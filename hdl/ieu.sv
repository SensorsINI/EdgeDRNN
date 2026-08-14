`default_nettype none

/**
 * Input Encoding Unit (IEU).
 *
 * The IEU implements the DeltaGRU temporal sparsity rule from paper Eq. (2):
 * it subtracts each stored reference value (x_hat or h_hat) from the current
 * value, compares the absolute delta with threshold Theta, and emits only
 * deltas that cross the threshold.  A non-zero delta is paired with a global
 * weight-column index so the top level can request exactly that column from
 * external memory (paper Fig. 4).
 *
 * State storage is split into:
 *   - current h banks, containing the latest hidden state for every layer;
 *   - previous/reference banks, containing x_hat and h_hat values that update
 *     only when their corresponding delta is emitted.
 *
 * The value and index streams use separate FIFOs but are admitted atomically:
 * neither half of a pair can advance unless the other half has space.  TLAST
 * boundary markers are preserved even for an all-zero delta vector, allowing
 * the HPC to distinguish delta-x and delta-h phases without a fake MAC.
 */
module ieu #(
    parameter int unsigned NUM_PE             = 16,
    parameter int unsigned ACT_INT_BW         = 8,
    parameter int unsigned ACT_FRA_BW         = 8,
    parameter int unsigned CFG_BW             = 32,
    parameter int unsigned CFG_NUM            = 32,
    parameter int unsigned DTH_BW             = 10,
    parameter int unsigned NZI_BW             = 16,
    parameter int unsigned NZL_FIFO_DEPTH     = 32,
    parameter int unsigned NUM_LAYER_BW       = 2,
    parameter int unsigned LAYER_SIZE_BW      = 9,
    parameter int unsigned MEM_STATE_DEPTH_BW = 5
    )(
    input  wire logic                                        clk,
    input  wire logic                                        rst_n,
    output logic unsigned [NUM_LAYER_BW:0]              curr_layer,
    output logic unsigned [LAYER_SIZE_BW:0]             curr_hid_size,
    input  wire logic          [CFG_BW-1:0]                  cfg            [CFG_NUM-1:0 ],

    // S_AXIS Interface (Input)                
    input  wire logic                                        s_inp_axis_tvalid,
    output logic                                        s_inp_axis_tready,
    input  wire logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] s_inp_axis_tdata,
    input  wire logic                                        s_inp_axis_tlast,

    // S_AXIS Interface (Feedback)             
    input  wire logic                                        s_fbk_axis_tvalid,
    output logic                                        s_fbk_axis_tready,
    input  wire logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] s_fbk_axis_tdata,
    input  wire logic                                        s_fbk_axis_tlast,

    // H(t-1) Access Interface
    input  wire logic                                        rd_en_hpt,
    output logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] dout_hpt,

    // NZVL AXIS Interface                     
    output logic                                        m_nzv_axis_tvalid,
    input  wire logic                                        m_nzv_axis_tready,
    output logic signed   [(ACT_INT_BW+ACT_FRA_BW)-1:0] m_nzv_axis_tdata,
    output logic                                        m_nzv_axis_tlast,

    // NZIL AXIS Interface                     
    output logic                                        m_nzi_axis_tvalid,
    input  wire logic                                        m_nzi_axis_tready,
    output logic signed   [NZI_BW-1:0]                  m_nzi_axis_tdata,
    output logic                                        m_nzi_axis_tlast
    );

    timeunit 1ns;
    timeprecision 1ps;
    import edgedrnn_pkg::*;

    //-----------------------
    //-- Local Parameter
    //-----------------------
    localparam int unsigned ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam int unsigned NUM_PE_BW = clog2_safe(NUM_PE);
    localparam int unsigned MAX_LAYER_SIZE = 1 << LAYER_SIZE_BW;
    localparam int unsigned MEM_STATE_DEPTH = MAX_LAYER_SIZE / NUM_PE;
    localparam int unsigned MAX_NUM_LAYER = 1 << NUM_LAYER_BW;
    typedef logic unsigned [LAYER_SIZE_BW:0] layer_size_t;
    typedef logic unsigned [NZI_BW-1:0] nzi_t;
    localparam logic [NUM_PE_BW-1:0] LAST_PE = NUM_PE_BW'(NUM_PE - 1);
    localparam logic [NUM_PE_BW-1:0] PRELAST_PE = NUM_PE_BW'(NUM_PE - 2);
    localparam logic [MEM_STATE_DEPTH_BW:0] PREV_H_OFFSET =
        (MEM_STATE_DEPTH_BW+1)'(MEM_STATE_DEPTH);

    /** Round a layer size to the PE-lane boundary without an integer-width cast. */
    function automatic layer_size_t round_layer_size(input layer_size_t value);
        layer_size_t lane_mask;
        lane_mask = layer_size_t'(NUM_PE - 1);
        return (value + lane_mask) & ~lane_mask;
    endfunction

    //-----------------------
    //-- Logic
    //-----------------------

    // Configurations
    layer_size_t input_size;
    layer_size_t r_input_size;
    layer_size_t hidden_size;
    layer_size_t r_hidden_size;
    layer_size_t cfg_layer_size [MAX_NUM_LAYER:0];
    logic unsigned [NUM_LAYER_BW:0]  num_layer;
    logic signed   [DTH_BW-1:0]      dth;
    logic signed   [DTH_BW-1:0]      thx;
    logic signed   [DTH_BW-1:0]      thh;
    logic signed   [ACT_BW-1:0]      dth_extended;

    // State Machine
    typedef enum logic [2:0] {
        IEU_IDLE = 3'h0,
        IEU_FB   = 3'h1,
        IEU_BIAS = 3'h2,
        IEU_DX   = 3'h3,
        IEU_DH   = 3'h4
    } ieu_state_t;

    // Controller Logic
    logic rst_n_cnt_addr_wr;
    logic rst_n_cnt_addr_rd;
    logic rst_n_cnt_rd_item_idx;
    logic rst_n_cnt_wr_item_idx;
    logic rst_n_cnt_layer;
    logic en_cnt_layer;

    // Delta Encoder Logic
    logic                              sel_dth;
    logic signed   [ACT_BW-1:0] r1_nzv;
    logic unsigned [NZI_BW-1:0] r1_nzi;
    logic unsigned [NZI_BW-1:0] layer_idx_offset;

    // AXIS FIFO Logic
    logic                                 nzvl_axis_tvalid;
    logic                                 nzvl_axis_tready;
    logic  signed        [ACT_BW-1:0]     nzvl_axis_tdata;
    logic                                 nzvl_axis_tlast;


    logic                                 nzil_axis_tvalid;
    logic                                 nzil_axis_tready;
    logic  signed        [NZI_BW-1:0]     nzil_axis_tdata;
    logic                                 nzil_axis_tlast;

    // NZVL/NZIL
    logic nzl_s_tvalid;
    logic nzi_s_tvalid;
    logic nzl_s_tready;
    logic nzl_pipe_tready;
    logic nzl_pair_tvalid;
    logic nzl_stage_tvalid;
    logic r_nzl_s_tlast;
    logic r1_nzl_s_tlast;
    logic nzl_s_tlast;
    logic null_nzl_s_tready;
    logic [ACT_BW-1:0] nzvl_s_tdata;
    logic [NZI_BW-1:0] nzil_s_tdata;

    // Layer Counter
    logic unsigned [NUM_LAYER_BW:0] cnt_layer;
    logic unsigned [NUM_LAYER_BW-1:0] active_layer_addr;
    logic unsigned [NUM_LAYER_BW-1:0] input_layer_addr;
    assign active_layer_addr = cnt_layer[NUM_LAYER_BW-1:0] - 1'b1;
    assign input_layer_addr  = cnt_layer[NUM_LAYER_BW-1:0] - NUM_LAYER_BW'(2);

    // State Memory
    logic                                   mem_curr_wr_en;
    logic                                   mem_curr_rd_en;
    logic unsigned [NUM_LAYER_BW-1:0]       mem_curr_l_wr_addr;
    logic unsigned [NUM_LAYER_BW-1:0]       mem_curr_l_rd_addr;
    logic unsigned [MEM_STATE_DEPTH_BW-1:0] mem_curr_wr_addr;
    logic unsigned [MEM_STATE_DEPTH_BW-1:0] mem_curr_rd_addr;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] mem_curr_din;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] mem_curr_dout;
    logic          [NUM_PE-1:0]             mem_prev_wr_en;
    logic                                   mem_prev_rd_en;
    logic unsigned [NUM_LAYER_BW-1:0]       mem_prev_l_wr_addr;
    logic unsigned [NUM_LAYER_BW-1:0]       mem_prev_l_rd_addr;
    logic unsigned [MEM_STATE_DEPTH_BW:0]   mem_prev_wr_addr;
    logic unsigned [MEM_STATE_DEPTH_BW:0]   mem_prev_rd_addr;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] mem_prev_dout;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] r_mem_prev_din;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] r1_mem_prev_din;

    //
    logic                                 s_mux_axis_sel;
    logic                                 s_mux_axis_tvalid;
    logic                                 s_mux_axis_tready;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] s_mux_axis_tdata  ;
    logic                                 s_mux_axis_tlast;

    // Item Read Index Counter
    logic unsigned [NZI_BW-1:0]    cnt_rd_item_idx;
    logic unsigned [NUM_PE_BW-1:0] cnt_rd_addr_batch;

    // Item Write Index Counter
    logic unsigned [NZI_BW-1:0]    cnt_wr_item_idx;
    logic unsigned [NUM_PE_BW-1:0] cnt_wr_addr_batch;

    //-----------------------
    //-- Connections
    //-----------------------

    // Assign memory connection
    assign dout_hpt = mem_curr_dout;
    assign mem_curr_din = s_mux_axis_tdata;

    assign m_nzv_axis_tvalid   = nzvl_axis_tvalid;
    assign nzvl_axis_tready = m_nzv_axis_tready;
    assign m_nzv_axis_tdata    = nzvl_axis_tdata;
    assign m_nzv_axis_tlast    = nzvl_axis_tlast;

    assign m_nzi_axis_tvalid   = nzil_axis_tvalid;
    assign nzil_axis_tready = m_nzi_axis_tready;
    assign m_nzi_axis_tdata    = nzil_axis_tdata;
    assign m_nzi_axis_tlast    = nzil_axis_tlast;

    assign curr_layer = cnt_layer;
    assign curr_hid_size = r_hidden_size;

    // Extend vector lengths once, then keep every item-counter comparison at
    // NZI_BW.  This makes wrap/truncation behavior explicit and lint-friendly.
    nzi_t input_length;
    nzi_t hidden_length;
    nzi_t input_last_index;
    nzi_t input_hidden_last_index;
    assign input_length = nzi_t'(r_input_size);
    assign hidden_length = nzi_t'(r_hidden_size);
    assign input_last_index = input_length - 1'b1;
    assign input_hidden_last_index = input_length + hidden_length - 1'b1;

    // Layer Counter
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            cnt_layer <= 0;
        end else if (!rst_n_cnt_layer) begin
            cnt_layer <= 1;
        end else if (en_cnt_layer) begin
            cnt_layer <= cnt_layer + 1;
        end
    end

    //-----------------------
    //-- Configuration
    //-----------------------

    // Configuration map is shared with software through edgedrnn_pkg.  Entry
    // CFG_IDX_INPUT_SIZE is the network input width; subsequent entries are
    // the hidden widths for layers 0..N-1.
    assign thx[DTH_BW-1:0] = cfg[CFG_IDX_THX][DTH_BW-1:0];
    assign thh[DTH_BW-1:0] = cfg[CFG_IDX_THH][DTH_BW-1:0];
    assign num_layer       = cfg[CFG_IDX_NUM_LAYER][NUM_LAYER_BW:0];
    assign dth_extended = {{(ACT_BW-DTH_BW){dth[DTH_BW-1]}}, dth};

    // Assign Layer Size
    always_comb begin
        for (int unsigned i=0; i<=MAX_NUM_LAYER; i=i+1) begin
            cfg_layer_size[i][LAYER_SIZE_BW:0] = cfg[CFG_IDX_INPUT_SIZE + i][LAYER_SIZE_BW:0];
        end
        if (cnt_layer == 0) begin  // Initial State
            input_size  = '0;
            hidden_size = '0;
        end else begin // If not reaching the last layer
            input_size  = cfg_layer_size[cnt_layer-1];
            hidden_size = cfg_layer_size[cnt_layer];
        end
    end

    // Buffer layer sizes for the state-machine pipeline.  The external input
    // may not fill an entire processing-element beat, so pad its logical size
    // to the next NUM_PE boundary.  Hidden layers are exported already padded.
    always_ff @ (posedge clk) begin
        if (cnt_layer == 0 || cnt_layer == 1) begin
            r_input_size <= round_layer_size(input_size);
        end else begin
            r_input_size <= input_size;
        end
        r_hidden_size <= hidden_size;
    end

    // Outer Offset RegFile Address Counter
    logic unsigned [NZI_BW-1:0] cnt_addr_outer_offset;
    logic en_cnt_addr_outer_offset;
    always_ff @ (posedge clk) begin
        if (!rst_n || !rst_n_cnt_layer) begin
            cnt_addr_outer_offset <= '0;
        end else if (en_cnt_addr_outer_offset) begin
            cnt_addr_outer_offset <= cnt_addr_outer_offset
                                   + nzi_t'(input_size)
                                   + nzi_t'(hidden_size)
                                   + 1'b1;
        end
    end
    assign layer_idx_offset = cnt_addr_outer_offset;

    // Slave AXIS Interface MUX
    always_comb begin
        if (!s_mux_axis_sel) begin
            s_mux_axis_tvalid = s_inp_axis_tvalid;
            s_inp_axis_tready = s_mux_axis_tready;
            s_fbk_axis_tready = 1'b1;
            s_mux_axis_tdata  = s_inp_axis_tdata;
            s_mux_axis_tlast  = s_inp_axis_tlast;
        end else begin
            s_mux_axis_tvalid = s_fbk_axis_tvalid;
            s_fbk_axis_tready = s_mux_axis_tready;
            s_inp_axis_tready = 1'b0;
            s_mux_axis_tdata  = s_fbk_axis_tdata;
            s_mux_axis_tlast  = s_fbk_axis_tlast;
        end
    end

    // Batch Write Address Counter
    logic en_bias, r_en_bias, r1_en_bias;
    logic item_valid, r_item_valid, r1_item_valid;

    // Update x_hat/h_hat only for a lane whose threshold comparison emitted a
    // non-zero delta.  This is the conditional reference update in Eq. (2).
    always_comb begin
        for (int unsigned i=0; i<NUM_PE; i=i+1) begin
            mem_prev_wr_en[i] = '0;
        end
        if (r1_item_valid && r1_nzv != 0) begin
            mem_prev_wr_en[cnt_wr_addr_batch] = 1'b1;
        end else begin
            mem_prev_wr_en[cnt_wr_addr_batch] = 1'b0;
        end
    end

    // Write Address Counter
    logic unsigned [MEM_STATE_DEPTH_BW-1:0] cnt_wr_addr;
    logic en_cnt_wr_addr;

    always_ff @ (posedge clk) begin
        if (!rst_n | !rst_n_cnt_addr_wr) begin
            cnt_wr_addr <= '0;
        end else if (en_cnt_wr_addr) begin
            cnt_wr_addr <= cnt_wr_addr + 1;
        end
    end

    // Read Address Counter
    logic unsigned [MEM_STATE_DEPTH_BW-1:0] cnt_rd_addr;
    logic en_cnt_rd_addr;

    always_ff @ (posedge clk) begin
        if (!rst_n | !rst_n_cnt_addr_rd) begin
            cnt_rd_addr <= '0;
        end else if (en_cnt_rd_addr) begin
            cnt_rd_addr <= cnt_rd_addr + 1;
        end
    end

    // Global column index.  Delta-h indices begin after the current layer's
    // delta-x range; cnt_addr_outer_offset skips all preceding layer columns.
    logic unsigned [NZI_BW-1:0]    cnt_nzi;
    logic                           rst_n_cnt_nzi;
    logic                           rst_n_cnt_nzi_sw;

    always_ff @ (posedge clk) begin
        if (!rst_n | !rst_n_cnt_nzi) begin
            cnt_nzi <= '0;
        end else if (!rst_n_cnt_nzi_sw) begin
            cnt_nzi <= nzi_t'(input_size);
        end else if (item_valid & nzl_pipe_tready) begin
            cnt_nzi <= cnt_nzi + 1;
        end
    end

    // Item Read Index Counter
    assign cnt_rd_addr_batch[NUM_PE_BW-1:0] = cnt_rd_item_idx[NUM_PE_BW-1:0];
    always_ff @ (posedge clk) begin
        if (!rst_n | !rst_n_cnt_rd_item_idx) begin
            cnt_rd_item_idx <= '0;
        end else if (item_valid & nzl_pipe_tready) begin
            cnt_rd_item_idx <= cnt_rd_item_idx + 1;
        end
    end

    // Item Write Index Counter
    assign cnt_wr_addr_batch[NUM_PE_BW-1:0] = cnt_wr_item_idx[NUM_PE_BW-1:0];

    always_ff @ (posedge clk) begin
        if (!rst_n | !rst_n_cnt_wr_item_idx) begin
            cnt_wr_item_idx <= '0;
        end else if (r1_item_valid & nzl_pipe_tready) begin
            cnt_wr_item_idx <= cnt_wr_item_idx + 1;
        end
    end

    // Memory Connections
    logic unsigned [MEM_STATE_DEPTH_BW:0] prev_wr_addr_offset;
    logic unsigned [MEM_STATE_DEPTH_BW:0] prev_rd_addr_offset;

    assign mem_curr_wr_addr = cnt_wr_addr;
    assign mem_curr_rd_addr = cnt_rd_addr;
    assign mem_prev_wr_addr = cnt_wr_addr + prev_wr_addr_offset;
    assign mem_prev_rd_addr = cnt_rd_addr + prev_rd_addr_offset;

    //-----------------
    //- Delta Encoder
    //-----------------
    logic signed   [MAX_NUM_LAYER:0][ACT_BW-1:0] bias_act0;
    logic signed   [MAX_NUM_LAYER:0][ACT_BW-1:0] bias_act1;
    logic signed   [ACT_BW-1:0] add_op0;
    logic signed   [ACT_BW-1:0] add_op1;
    logic signed   [ACT_BW-1:0] add_out;
    logic signed   [ACT_BW-1:0] r_nzv;
    logic signed   [NZI_BW-1:0] r_nzi;
    logic en_de;
    logic r_en_de;
    logic [1:0] sel_de;

    // Bias Registers
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            for (int unsigned i = 0; i <= MAX_NUM_LAYER; i = i + 1) begin
            bias_act0[i] <= 2**ACT_FRA_BW;
            bias_act1[i] <= '0;
            end
        end else if (en_bias) begin
            bias_act1[curr_layer] <= bias_act0[curr_layer];
        end
    end

    // Delta encoder pipeline stage 0.  sel_de chooses a synthetic bias delta,
    // external x minus x_hat, or current h minus h_hat.  The matching global
    // column index is registered beside the operands.
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            r_en_de <= 1'b0;
            add_op0 <= '0;
            add_op1 <= '0;
            r_nzi   <= '0;
        end else begin
            if (nzl_pipe_tready) begin
                r_en_de <= en_de;
            end
            if (sel_de == 2'd0 && en_de && nzl_pipe_tready) begin
                add_op0 <= bias_act0[curr_layer];
                add_op1 <= bias_act1[curr_layer];
                r_nzi   <= cnt_addr_outer_offset;
            end else if (sel_de == 2'd1 && en_de && nzl_pipe_tready) begin
                add_op0 <= s_mux_axis_tdata[cnt_rd_addr_batch];
                add_op1 <= mem_prev_dout[cnt_rd_addr_batch];
                r_nzi   <= cnt_nzi + layer_idx_offset + 1;
            end else if (en_de && nzl_pipe_tready) begin
                add_op0 <= mem_curr_dout[cnt_rd_addr_batch];
                add_op1 <= mem_prev_dout[cnt_rd_addr_batch];
                r_nzi   <= cnt_nzi + layer_idx_offset + 1;
            end
        end
    end

    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            r_en_bias      <= '0;
            r_item_valid   <= '0;
            r_nzl_s_tlast <= '0;
        end else if (en_de && nzl_pipe_tready) begin
            r_en_bias      <= en_bias;
            r_item_valid   <= item_valid;
            r_nzl_s_tlast <= nzl_s_tlast;
        end
    end

    // Threshold stage: emit the exact signed difference at or beyond Theta;
    // otherwise emit zero.  Equality is included, matching Eq. (2).
    always_comb begin
        add_out = $signed(add_op0) - $signed(add_op1);
        // If above threshold
        if (add_out >= dth_extended || add_out <= -dth_extended) begin
            r_nzv = add_out;
        // If below threshold
        end else begin
            r_nzv = '0;
        end
    end

    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            dth <= '0;
        end else if (sel_dth) begin
            dth <= thh;
        end else begin
            dth <= thx;
        end
    end

    // Pairing stage.  A real delta consumes one slot in both FIFOs.  A TLAST
    // marker with no delta consumes only the value FIFO, which lets the HPC
    // observe an empty phase without issuing a meaningless weight request.
    assign nzvl_s_tdata = r1_nzv;
    assign nzil_s_tdata = (r1_nzv == 0) ? 0 : r1_nzi;

    assign nzl_pair_tvalid = r1_nzv != 0 && (r1_item_valid | r1_en_bias);
    assign nzl_stage_tvalid = nzl_pair_tvalid | r1_nzl_s_tlast;
    assign nzl_s_tvalid = nzl_stage_tvalid & (~nzl_pair_tvalid | null_nzl_s_tready);
    assign nzi_s_tvalid = nzl_pair_tvalid & nzl_s_tready;
    assign nzl_pipe_tready = ~nzl_stage_tvalid | (nzl_s_tready & (~nzl_pair_tvalid | null_nzl_s_tready));

    always_ff @ (posedge clk) begin
        if (!rst_n) begin 
            r1_nzv         <= '0;
            r1_nzi         <= '0;
            r1_en_bias     <= '0;
            r1_item_valid  <= '0;
            r1_nzl_s_tlast <= '0;
        end else if (r_en_de & nzl_pipe_tready) begin
            r1_nzv         <= r_nzv;
            r1_nzi         <= r_nzi;
            r1_en_bias     <= r_en_bias;
            r1_item_valid  <= r_item_valid;
            r1_nzl_s_tlast <= r_nzl_s_tlast;
        end
    end

    // Delta FIFOs
    axis_fifo #(
        .AXIS_BUS_WIDTH (ACT_BW),
        .DEPTH_WIDTH    (clog2_safe(NZL_FIFO_DEPTH))
    ) i_nzvl (
        .m_axi_aclk    (clk),
        .m_axi_aresetn (rst_n),
        .s_axis_tdata  (nzvl_s_tdata),
        .s_axis_tvalid (nzl_s_tvalid),
        .s_axis_tlast  (r1_nzl_s_tlast),
        .s_axis_tready (nzl_s_tready),
        .m_axis_tdata  (nzvl_axis_tdata),
        .m_axis_tvalid (nzvl_axis_tvalid),
        .m_axis_tlast  (nzvl_axis_tlast),
        .m_axis_tready (nzvl_axis_tready)
    );

    axis_fifo #(
        .AXIS_BUS_WIDTH (NZI_BW),
        .DEPTH_WIDTH    (clog2_safe(NZL_FIFO_DEPTH))
    ) i_nzil (
        .m_axi_aclk    (clk),
        .m_axi_aresetn (rst_n),
        .s_axis_tdata  (nzil_s_tdata),
        .s_axis_tvalid (nzi_s_tvalid),
        .s_axis_tlast  (r1_nzl_s_tlast),
        .s_axis_tready (null_nzl_s_tready),
        .m_axis_tdata  (nzil_axis_tdata),
        .m_axis_tvalid (nzil_axis_tvalid),
        .m_axis_tlast  (nzil_axis_tlast),
        .m_axis_tready (nzil_axis_tready)
    );

    // State Machine Sequential Logic
    (* mark_debug = "true" *) ieu_state_t ieu_cs, ieu_ns;
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            ieu_cs <= IEU_IDLE;
        end else begin
            ieu_cs <= ieu_ns;
        end
    end

    // State Machine Combinational Logic
    always_comb begin
        rst_n_cnt_addr_wr        = '1;
        rst_n_cnt_addr_rd        = '1;
        rst_n_cnt_rd_item_idx    = '1;
        rst_n_cnt_wr_item_idx    = '1;
        rst_n_cnt_layer          = '1;
        rst_n_cnt_nzi            = '1;
        rst_n_cnt_nzi_sw         = '1;
        s_mux_axis_tready           = '0;
        en_bias                  = '0;
        item_valid               = '0;
        en_cnt_layer             = '0;
        nzl_s_tlast              = '0;
        en_de                    = '0;
        sel_de                   = '0;
        s_mux_axis_sel              = '0;
        en_cnt_addr_outer_offset = '0;
        prev_wr_addr_offset      = '0;
        prev_rd_addr_offset      = '0;
        en_cnt_wr_addr           = (cnt_wr_addr_batch == LAST_PE && nzl_pipe_tready == 1);
        en_cnt_rd_addr           = '0;
        mem_curr_wr_en           = '0;
        mem_curr_rd_en           = '0;
        mem_prev_rd_en           = '0;
        mem_curr_l_wr_addr       = active_layer_addr;
        mem_curr_l_rd_addr       = active_layer_addr;
        mem_prev_l_wr_addr       = active_layer_addr;
        mem_prev_l_rd_addr       = active_layer_addr;
        sel_dth = '0;
        //en_nzi_valid = '0;
        ieu_ns = IEU_IDLE;
        case(ieu_cs)
            // Wait for the first external input vector.  All per-layer and
            // per-vector counters are held at their starting positions.
            IEU_IDLE: //------------------------------------------------
            begin
            ieu_ns                = IEU_IDLE;
            rst_n_cnt_addr_rd     = '0;
            rst_n_cnt_addr_wr     = '0;
            rst_n_cnt_rd_item_idx = '0;
            rst_n_cnt_wr_item_idx = '0;
            rst_n_cnt_nzi         = '0;
            if (s_mux_axis_tvalid == 1'b1) begin
                ieu_ns             = IEU_FB;
            end
            end
            
            // Consume the activation feedback from the previous HPC pass and
            // store it as the current hidden state.  For layer zero there is
            // no preceding feedback, so proceed directly to the bias marker.
            IEU_FB: //------------------------------------------------
            begin
            ieu_ns = IEU_FB;
            s_mux_axis_sel     = '1;
            if (cnt_layer == 0) begin // First hidden layer
                ieu_ns = IEU_BIAS;
                en_cnt_layer = '1;
            end else begin // Other hidden layers
                s_mux_axis_tready  = '1;
                mem_curr_rd_en  = rd_en_hpt;
                en_cnt_rd_addr  = mem_curr_rd_en;
                mem_curr_wr_en  = s_mux_axis_tready & s_mux_axis_tvalid;
                en_cnt_wr_addr  = mem_curr_wr_en;
                
                if (s_mux_axis_tlast & s_mux_axis_tvalid) begin
                    ieu_ns = IEU_BIAS;
                    rst_n_cnt_addr_wr = '0;
                    rst_n_cnt_addr_rd = '0;
                    if (cnt_layer == num_layer) begin
                        rst_n_cnt_layer = '0;
                    end else begin
                        en_cnt_layer = '1;
                    end
                end
            end   
            end
        
            // Emit the per-layer bias column first.  On later layers, prime
            // both the current-state and reference-state synchronous reads.
            IEU_BIAS: //------------------------------------------------
            begin
            ieu_ns     = IEU_BIAS;
            en_bias    = nzl_pipe_tready;
            en_de      = '1;
            mem_curr_l_rd_addr = input_layer_addr;
            
            if (en_bias) begin
                ieu_ns  = IEU_DX;
            end
            
            if (cnt_layer == 1) begin // First hidden layer
                mem_prev_rd_en = '1;
            end else begin // Other hidden layers
                mem_curr_rd_en = '1;
                mem_prev_rd_en = '1;
            end
            end
            
            // Encode delta-x: external input for layer one, otherwise the
            // preceding layer's freshly calculated hidden activation.
            IEU_DX: //------------------------------------------------
            begin
            ieu_ns         = IEU_DX;
            en_de          = '1;
            sel_dth        = '0;
            en_cnt_rd_addr = (cnt_rd_addr_batch == PRELAST_PE && nzl_pipe_tready == 1);
            en_cnt_wr_addr = (cnt_wr_addr_batch == LAST_PE && nzl_pipe_tready == 1);
            mem_curr_l_rd_addr = input_layer_addr; // Previous layer supplies x for this layer.
            
            if (cnt_layer == 1) begin // First hidden layer
                item_valid = s_mux_axis_tvalid;
                sel_de = 1;
                s_mux_axis_tready = (cnt_rd_addr_batch == LAST_PE && nzl_pipe_tready == 1);
                mem_prev_rd_en = s_mux_axis_tready;
            end else begin // Other hidden layers
                item_valid = '1;
                sel_de = 2;
                mem_curr_rd_en = (cnt_rd_addr_batch == LAST_PE && nzl_pipe_tready == 1);
                mem_prev_rd_en = mem_curr_rd_en;
            end
            
            if (cnt_rd_item_idx == input_last_index && nzl_pipe_tready == 1) begin
                nzl_s_tlast       = '1;
            //   en_nzi_valid      = '0;
            end
            
            if (cnt_rd_item_idx >= input_last_index && nzl_pipe_tready == 1) begin
                rst_n_cnt_addr_rd = '0;
                mem_curr_rd_en    = '0;
                mem_prev_rd_en    = '0;
            end
            
            if (cnt_rd_item_idx > input_last_index && nzl_pipe_tready == 1) begin
                item_valid        = '0;
            end
            
            if (cnt_wr_item_idx == input_last_index && nzl_pipe_tready == 1) begin
                ieu_ns  = IEU_DH;
                sel_dth = '1;
                rst_n_cnt_nzi_sw  = '0;
                rst_n_cnt_addr_wr = '0;
                mem_curr_rd_en    = '1;
                mem_prev_rd_en    = '1;
                mem_curr_l_rd_addr = active_layer_addr;
                // Encode Address
                prev_rd_addr_offset  = PREV_H_OFFSET;
            end
            
            end
            
            // Encode delta-h from current h(t-1) and h_hat.  The reference
            // address offset selects the h_hat half of previous-state memory.
            IEU_DH: //------------------------------------------------
            begin
            ieu_ns              = IEU_DH;
            sel_dth             = '1;
            item_valid          = '1;
            sel_de              = 2;
            en_de               = nzl_pipe_tready;
            en_cnt_rd_addr      = (cnt_rd_addr_batch == PRELAST_PE && nzl_pipe_tready == 1);
            en_cnt_wr_addr      = (cnt_wr_addr_batch == LAST_PE && nzl_pipe_tready == 1);
            prev_rd_addr_offset = PREV_H_OFFSET;
            prev_wr_addr_offset = PREV_H_OFFSET;
            
            mem_curr_rd_en      = (cnt_rd_addr_batch == LAST_PE && nzl_pipe_tready == 1);
            mem_prev_rd_en      = mem_curr_rd_en;
            
            if (cnt_rd_item_idx == input_hidden_last_index && nzl_pipe_tready == 1) begin
                nzl_s_tlast              = '1;
                en_cnt_addr_outer_offset = '1;
            end
            
            if (cnt_rd_item_idx >= input_hidden_last_index && nzl_pipe_tready == 1) begin
                rst_n_cnt_addr_rd        = '0;
                mem_curr_rd_en           = '0;
                mem_prev_rd_en           = '0;
            end
            
            if (cnt_rd_item_idx > input_hidden_last_index && nzl_pipe_tready == 1) begin
                item_valid = '0;
            end
            
            if (cnt_wr_item_idx == input_hidden_last_index && nzl_pipe_tready == 1) begin
                ieu_ns                   = IEU_FB;
                rst_n_cnt_rd_item_idx    = '0;
                rst_n_cnt_wr_item_idx    = '0;
                rst_n_cnt_nzi            = '0;
                rst_n_cnt_addr_wr        = '0;
                mem_curr_rd_en           = '0;
                mem_prev_rd_en           = '0;
            end
            end
            
            default: //-------------------------------------------
            begin
            rst_n_cnt_addr_wr        = '1;
            rst_n_cnt_addr_rd        = '1;
            rst_n_cnt_rd_item_idx    = '1;
            rst_n_cnt_wr_item_idx    = '1;
            rst_n_cnt_layer          = '1;
            rst_n_cnt_nzi            = '1;
            rst_n_cnt_nzi_sw         = '1;
            s_mux_axis_tready        = '0;
            en_bias                  = '0;
            item_valid               = '0;
            en_cnt_layer             = '0;
            nzl_s_tlast              = '0;
            en_de                    = '0;
            sel_de                   = '0;
            s_mux_axis_sel           = '0;
            en_cnt_addr_outer_offset = '0;
            prev_wr_addr_offset      = '0;
            prev_rd_addr_offset      = '0;
            en_cnt_wr_addr           = '0;
            en_cnt_rd_addr           = '0;
            mem_curr_wr_en           = '0;
            mem_curr_rd_en           = '0;
            mem_prev_rd_en           = '0;
            mem_curr_l_wr_addr       = active_layer_addr;
            mem_curr_l_rd_addr       = active_layer_addr;
            mem_prev_l_wr_addr       = active_layer_addr;
            mem_prev_l_rd_addr       = active_layer_addr;
            //en_nzi_valid = '0;
            ieu_ns                   = IEU_IDLE;
            end
        endcase
    end

    // Two-cycle alignment from selected source data to the conditional
    // reference-memory write enable generated by the delta pipeline.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_mem_prev_din  <= '0;
            r1_mem_prev_din <= '0;
        end else if (sel_de == 2'd1 & nzl_pipe_tready) begin
            r_mem_prev_din <= s_mux_axis_tdata;
            r1_mem_prev_din <= r_mem_prev_din;
        end else if (item_valid & nzl_pipe_tready) begin
            r_mem_prev_din <= mem_curr_dout;
            r1_mem_prev_din <= r_mem_prev_din;
        end
    end

    // Banked current/reference state memories from paper Fig. 4.
    mem_state # (
        .NUM_PE             (NUM_PE            ),
        .ACT_INT_BW         (ACT_INT_BW        ),
        .ACT_FRA_BW         (ACT_FRA_BW        ),
        .NUM_LAYER_BW       (NUM_LAYER_BW      ),
        .MEM_STATE_DEPTH_BW (MEM_STATE_DEPTH_BW)
    ) i_mem_state (
        .clk            (clk),
    `ifdef SIM_DEBUG //----------------------------------------------
        .rst_n          (rst_n),
    `endif //--------------------------------------------------------
        .curr_wr_en     (mem_curr_wr_en  ),
        .curr_rd_en     (mem_curr_rd_en  ),
        .curr_l_wr_addr (mem_curr_l_wr_addr),
        .curr_l_rd_addr (mem_curr_l_rd_addr),
        .curr_wr_addr   (mem_curr_wr_addr),
        .curr_rd_addr   (mem_curr_rd_addr),
        .curr_din       (mem_curr_din    ),
        .curr_dout      (mem_curr_dout   ),
        .prev_wr_en     (mem_prev_wr_en  ),
        .prev_rd_en     (mem_prev_rd_en  ),
        .prev_l_wr_addr (mem_prev_l_wr_addr),
        .prev_l_rd_addr (mem_prev_l_rd_addr),
        .prev_wr_addr   (mem_prev_wr_addr),
        .prev_rd_addr   (mem_prev_rd_addr),
        .prev_din       (r1_mem_prev_din    ),
        .prev_dout      (mem_prev_dout   )
    );
`ifndef SYNTHESIS
    initial begin
        assert (NUM_PE >= 2 && is_power_of_two(NUM_PE))
            else $fatal(1, "NUM_PE must be a power of two and at least two");
        assert (MAX_LAYER_SIZE % NUM_PE == 0)
            else $fatal(1, "Maximum layer size must be divisible by NUM_PE");
        assert (is_power_of_two(NZL_FIFO_DEPTH))
            else $fatal(1, "NZL_FIFO_DEPTH must be a power of two");
        assert (CFG_NUM > CFG_IDX_HIDDEN_BASE)
            else $fatal(1, "Configuration register file is too small");
    end
`endif

endmodule

`default_nettype wire

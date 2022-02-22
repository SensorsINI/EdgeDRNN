    //-----------------------------------------------------
    // Weight Memory for Simulation
    // Design Name : gen_weight
    // File Name   : gen_weight.sv
    //-----------------------------------------------------
    `timescale 1ns/1ps

    module gen_weight # (
    NUM_PE        = 8,
    W_BW          = 8,
    LAYER_SIZE_BW = 10
    )(
    input  logic                   s_axi_aclk,
    input  logic                   s_axi_aresetn,
    input  logic                   s_axis_tvalid,
    output logic                   s_axis_tready,
    input  logic [80-1:0]          s_axis_tdata,
    input  logic                   s_axis_tlast,         
    output logic                   m_axis_tvalid,
    input  logic                   m_axis_tready,
    output logic [NUM_PE*W_BW-1:0] m_axis_tdata,
    output logic                   m_axis_tlast
    );

    //-----------------------
    //-- Logic
    //-----------------------
    longint burst_len;
    longint burst_len_bytes;
    logic ld_ptr_n;
    logic [NUM_PE*W_BW-1:0] mem [];
    longint cnt_rd_beat;
    longint cnt_rd_addr;
    logic   cnt_rd_addr_en;
    longint rd_addr;

    // State Machine
    typedef enum logic [1:0]   {WMEM_IDLE = 2'h0,
                                WMEM_READ = 2'h1,
                                WMEM_VOID = 2'h2} WMEM_STATE;
    WMEM_STATE wmem_cs, wmem_ns;

    //-----------------------
    //-- Controller
    //-----------------------
    always_comb begin
        m_axis_tdata = mem[cnt_rd_addr];
        burst_len_bytes = s_axis_tdata[22:0];
        burst_len = burst_len_bytes/NUM_PE;
        rd_addr = s_axis_tdata[63:32]/NUM_PE;
    end

    // Read Beat Counter
    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            cnt_rd_addr <= '0;
        end else if (!ld_ptr_n) begin
            cnt_rd_addr <= rd_addr;
        end else if (cnt_rd_addr_en) begin
            cnt_rd_addr <= cnt_rd_addr + 1;
        end
    end

    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn | (!ld_ptr_n)) begin
            cnt_rd_beat <= '0;
        end else if (cnt_rd_addr_en) begin
            cnt_rd_beat <= cnt_rd_beat + 1;
        end
    end

    logic unsigned [31:0] seed;

    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            seed <= 0;
        end else begin
            seed <= seed + 1;
        end
    end

    // Sequencial Logic
    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            wmem_cs <= WMEM_IDLE;
        end else begin
            wmem_cs <= wmem_ns;
        end
    end

    // Combinational Logic
    always_comb begin
        wmem_ns        = WMEM_IDLE;
        ld_ptr_n       = '1;
        s_axis_tready  = '0;
        cnt_rd_addr_en = '0;
        m_axis_tvalid  = '0;
        m_axis_tlast   = '0;
        case(wmem_cs)
            // Initialization
            WMEM_IDLE: //------------------------------------------------
            begin
            wmem_ns = WMEM_IDLE;
            if (s_axis_tvalid) begin
                ld_ptr_n = '0;
                wmem_ns = WMEM_READ;
            end
            end
            
            // IDLE
            WMEM_READ: //------------------------------------------------
            begin
                wmem_ns = WMEM_READ;
                s_axis_tready = '0;
                m_axis_tvalid = $urandom(seed);
                // m_axis_tvalid = 1;
                cnt_rd_addr_en = m_axis_tvalid & m_axis_tready;
                
                if (cnt_rd_beat == burst_len-2 && m_axis_tready & m_axis_tvalid) begin
                    s_axis_tready = '1;
                end
                
                if (cnt_rd_beat == burst_len-1) begin
                    m_axis_tlast = '1;
                    if (m_axis_tready & m_axis_tvalid) begin
                        if (s_axis_tvalid) begin
                            ld_ptr_n = '0;
                        end else begin
                            ld_ptr_n = '1;
                            wmem_ns = WMEM_IDLE;
                        end
                    end
                end
            end
            
            default: //-------------------------------------------
            begin
            wmem_ns        = WMEM_IDLE;
            ld_ptr_n     = '1;
            s_axis_tready  = '0;
            cnt_rd_addr_en = '0;
            m_axis_tvalid  = '0;
            m_axis_tlast   = '0;
            end
        endcase
    end

    //-----------------------
    //-- Read File
    //-----------------------
    initial begin
        $readmemh("../../../../../../hdl/tb/edgedrnn_tb_params.txt", mem);
    end
endmodule

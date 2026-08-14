`default_nettype none

/**
 * EdgeDRNN accelerator top level.
 *
 * The block boundaries follow paper Fig. 3:
 *
 *   input/feedback -> IEU -> {non-zero value, column index}
 *                         -> AXI Datamover command + external weights
 *                         -> HPC/HPE -> next hidden-state activation
 *
 * The IEU implements the threshold comparison in Eq. (2) and generates sparse
 * whole-column updates.  The HPC applies those updates to the four DeltaGRU
 * memories in Eq. (3), then time-multiplexes the HPE lanes to form a new
 * hidden state.  Intermediate-layer activations feed back into the IEU; final
 * layer activations are also exposed on m_out_axis.
 *
 * All stream ports use AXI4-Stream ready/valid semantics: a beat transfers on
 * a rising edge only when both TVALID and TREADY are high.  The output fork
 * tracks the feedback and external consumers independently so either side may
 * apply backpressure without duplicating or dropping a beat.
 */
module edgedrnn #(
    parameter int unsigned C_S_AXI_DATA_WIDTH = 32,
    parameter int unsigned C_S_AXI_ADDR_WIDTH = 7,
    parameter int unsigned NUM_PE             = 8,
    parameter int unsigned ACT_INT_BW         = 8,
    parameter int unsigned ACT_FRA_BW         = 8,
    parameter int unsigned W_INT_BW           = 1,
    parameter int unsigned W_FRA_BW           = 7,
    parameter int unsigned NUM_LAYER_BW       = 2,
    parameter int unsigned LAYER_SIZE_BW      = 10,
    parameter int unsigned DTH_BW             = 10,
    parameter int unsigned NZI_BW             = 16,
    parameter int unsigned NZL_FIFO_DEPTH     = 32,
    parameter string       CREG_FILE          = "edgedrnn_tb_creg.txt"
    )(
    // Ports of Axi-Lite Slave Bus Interface S_AXI
    input  wire logic                              clk,
    input  wire logic                              rst_n,
    // Ports of Axi Slave Bus Interface S_AXI
    `ifndef SIM_DEBUG //----------------------------------------------
    input  wire logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire logic [2:0]                        s_axi_awprot,
    input  wire logic                              s_axi_awvalid,
    output logic                              s_axi_awready,
    input  wire logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire logic                              s_axi_wvalid,
    output logic                              s_axi_wready,
    output logic [1:0]                        s_axi_bresp,
    output logic                              s_axi_bvalid,
    input  wire logic                              s_axi_bready,
    input  wire logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire logic [2:0]                        s_axi_arprot,
    input  wire logic                              s_axi_arvalid,
    output logic                              s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output logic [1:0]                        s_axi_rresp,
    output logic                              s_axi_rvalid,
    input  wire logic                              s_axi_rready,
    `endif //--------------------------------------------------------
    // S_AXIS Interface (Input)               
    input  wire logic                                        s_inp_axis_tvalid,
    output logic                                        s_inp_axis_tready,
    input  wire logic [NUM_PE*(ACT_INT_BW + ACT_FRA_BW)-1:0] s_inp_axis_tdata,
    input  wire logic                                        s_inp_axis_tlast,
    // S_AXIS Interface (Weight)               
    input  wire logic                                        s_w_axis_tvalid,
    output logic                                        s_w_axis_tready,
    input  wire logic [NUM_PE*(W_INT_BW + W_FRA_BW)-1:0]     s_w_axis_tdata,
    input  wire logic                                        s_w_axis_tlast,
    // M_AXIS Interface (NZIL)            
    output logic                                        m_inst_axis_tvalid,
    input  wire logic                                        m_inst_axis_tready,
    output logic [edgedrnn_pkg::DATAMOVER_CMD_WIDTH-1:0] m_inst_axis_tdata,
    output logic                                        m_inst_axis_tlast,
    // Activation Output
    output logic                                        m_out_axis_tvalid,
    input  wire logic                                        m_out_axis_tready,
    output logic [NUM_PE*(ACT_INT_BW + ACT_FRA_BW)-1:0] m_out_axis_tdata,
    output logic                                        m_out_axis_tlast
    );

    timeunit 1ns;
    timeprecision 1ps;
    import edgedrnn_pkg::*;

    localparam int unsigned ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam int unsigned W_BW = W_INT_BW + W_FRA_BW;
    localparam int unsigned MAX_LAYER_SIZE = 1 << LAYER_SIZE_BW;
    localparam int unsigned MEM_STATE_DEPTH = MAX_LAYER_SIZE / NUM_PE;
    localparam int unsigned CFG_BW = 32;
    localparam int unsigned CFG_NUM = 32;
    localparam int unsigned NUM_PE_BW = clog2_safe(NUM_PE);
    localparam int unsigned MEM_STATE_DEPTH_BW = clog2_safe(MEM_STATE_DEPTH);
    localparam int unsigned MEM_ACC_DEPTH_BW =
        NUM_LAYER_BW + LAYER_SIZE_BW + 2 - NUM_PE_BW;
    localparam int unsigned MEM_ACC_DEPTH_SL_BW =
        LAYER_SIZE_BW + 2 - NUM_PE_BW;

    //-----------------------
    //-- Logic
    //-----------------------
    // IEU
    logic              nzi_axis_tvalid;
    logic [NZI_BW-1:0] nzi_axis_tdata;
    logic              nzi_axis_tlast;
    logic              nzi_axis_tready;

    logic          [CFG_BW-1:0]       cfg            [CFG_NUM-1:0 ];
    logic unsigned [NUM_LAYER_BW:0]   ieu_hpc_curr_layer;
    logic unsigned [LAYER_SIZE_BW:0]  curr_hid_size;



    // S_AXIS Interface (Feedback)
    logic                                   fbk_axis_tvalid;
    logic                                   fbk_axis_tready;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] fbk_axis_tdata;
    logic                                   fbk_axis_tlast;             

    // H(t-1) Access Interface
    logic                                   hpc_ieu_rd_en_hpt;
    logic signed   [NUM_PE-1:0][ACT_BW-1:0] ieu_hpc_dout_hpt;

    // NZVL AXIS Interface
    logic                             nzv_axis_tvalid;
    logic                             nzv_axis_tready;
    logic signed   [ACT_BW-1:0]       nzv_axis_tdata;
    logic                             nzv_axis_tlast;

    // Unpacked I/O     
    logic signed [NUM_PE-1:0][ACT_BW-1:0] roll_s_inp_axis_tdata ;
    logic signed [NUM_PE-1:0][W_BW-1:0]   roll_s_w_axis_tdata;
    // Packed I/O
    logic [NUM_PE*ACT_BW-1:0]   pk_m_out_axis_tdata;

    // HPC M_AXIS Interface
    logic                                 act_axis_tvalid;
    logic                                 act_axis_tready;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] act_axis_tdata ;
    logic                                 act_axis_tlast;
    logic                                 out_axis_done_reg;
    logic                                 fbk_axis_done_reg;
    logic                                 act_packet_active_reg;
    logic                                 act_packet_is_final_reg;
    logic                                 act_packet_is_final;

    //-----------------------
    //-- AXI Datamover CMD
    //-----------------------
    logic [15:0] bytes_per_burst;
    logic [31:0] w_addr;
    // One non-zero delta triggers a whole weight column.  A DeltaGRU column
    // contains the reset, update, and candidate weights, hence 3*hidden_size
    // bytes for the default INT8 weight format (paper Fig. 6).
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            w_addr          <= '0;
            bytes_per_burst <= '0;
        end else begin
            w_addr          <= cfg[CFG_IDX_W_BASEADDR];
            bytes_per_burst <= 3*curr_hid_size;
        end
    end

    // Xilinx AXI Datamover 80-bit MM2S command.  BTT is the byte count, the
    // start address selects the sparse column, and the remaining fields are
    // fixed attributes expected by the surrounding Vivado block design.
    assign m_inst_axis_tdata[22:0]  = 3*curr_hid_size;
    assign m_inst_axis_tdata[23]    = 1'b1;
    assign m_inst_axis_tdata[29:24] = 6'b0;
    assign m_inst_axis_tdata[30]    = 1'b1;
    assign m_inst_axis_tdata[31]    = 1'b0;
    assign m_inst_axis_tdata[63:32] = w_addr + nzi_axis_tdata*bytes_per_burst;
    assign m_inst_axis_tdata[67:64] = 4'b0110;
    assign m_inst_axis_tdata[71:68] = '0;
    assign m_inst_axis_tdata[75:72] = '1;
    assign m_inst_axis_tdata[79:76] = 4'b0010;
    assign m_inst_axis_tvalid = nzi_axis_tvalid;
    assign m_inst_axis_tlast  = nzi_axis_tlast;
    assign nzi_axis_tready = m_inst_axis_tready;

    // Configuration Registers
    `ifndef SIM_DEBUG //----------------------------------------------
    logic [C_S_AXI_DATA_WIDTH-1:0] reg0 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg1 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg2 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg3 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg4 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg5 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg6 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg7 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg8 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg9 ;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg10;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg11;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg12;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg13;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg14;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg15;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg16;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg17;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg18;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg19;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg20;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg21;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg22;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg23;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg24;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg25;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg26;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg27;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg28;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg29;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg30;
    logic [C_S_AXI_DATA_WIDTH-1:0] reg31;
    `endif //--------------------------------------------------------

    //-----------------------
    //-- I/O Packaging
    //-----------------------
    // External buses are flat vectors, while the datapath uses a packed array
    // indexed by processing-element lane.  Indexed part-selects make the lane
    // ordering explicit: lane 0 always occupies the least-significant bits.
    for (genvar pe = 0; pe < NUM_PE; pe++) begin : g_repack_io
        assign roll_s_inp_axis_tdata[pe] = s_inp_axis_tdata[pe*ACT_BW +: ACT_BW];
        assign roll_s_w_axis_tdata[pe]   = s_w_axis_tdata[pe*W_BW +: W_BW];
        assign pk_m_out_axis_tdata[pe*ACT_BW +: ACT_BW] = act_axis_tdata[pe];
    end

    //-----------------------
    //-- Configuration
    //-----------------------

    // Get Config
    logic unsigned [NUM_LAYER_BW:0]  num_layer;
    assign num_layer = cfg[CFG_IDX_NUM_LAYER][NUM_LAYER_BW:0];

    //-----------------------
    //-- Connections
    //-----------------------
    `ifndef SIM_DEBUG //----------------------------------------------
    assign cfg[0] = reg0;
    assign cfg[1] = reg1;
    assign cfg[2] = reg2;
    assign cfg[3] = reg3;
    assign cfg[4] = reg4;
    assign cfg[5] = reg5;
    assign cfg[6] = reg6;
    assign cfg[7] = reg7;
    assign cfg[8] = reg8;
    assign cfg[9] = reg9;
    assign cfg[10] = reg10;
    assign cfg[11] = reg11;
    assign cfg[12] = reg12;
    assign cfg[13] = reg13;
    assign cfg[14] = reg14;
    assign cfg[15] = reg15;
    assign cfg[16] = reg16;
    assign cfg[17] = reg17;
    assign cfg[18] = reg18;
    assign cfg[19] = reg19;
    assign cfg[20] = reg20;
    assign cfg[21] = reg21;
    assign cfg[22] = reg22;
    assign cfg[23] = reg23;
    assign cfg[24] = reg24;
    assign cfg[25] = reg25;
    assign cfg[26] = reg26;
    assign cfg[27] = reg27;
    assign cfg[28] = reg28;
    assign cfg[29] = reg29;
    assign cfg[30] = reg30;
    assign cfg[31] = reg31;
    `else  //--------------------------------------------------------
    initial begin
        $readmemh(CREG_FILE, cfg);
    end
    `endif //--------------------------------------------------------

    // Latch the destination for a complete layer-output packet.  The IEU layer
    // counter can advance as soon as its feedback copy of TLAST is accepted,
    // while the external output copy may still be stalled.  Using the live
    // counter in that situation would withdraw TVALID illegally.  Before the
    // first beat is registered, the live comparison supplies the same-cycle
    // routing decision; all later beats use the latched packet classification.
    assign act_packet_is_final = act_packet_active_reg
                               ? act_packet_is_final_reg
                               : (ieu_hpc_curr_layer == num_layer);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            act_packet_active_reg   <= 1'b0;
            act_packet_is_final_reg <= 1'b0;
        end else begin
            if (!act_packet_active_reg && act_axis_tvalid) begin
                act_packet_active_reg   <= 1'b1;
                act_packet_is_final_reg <= (ieu_hpc_curr_layer == num_layer);
            end
            if (act_axis_tvalid && act_axis_tready && act_axis_tlast) begin
                act_packet_active_reg <= 1'b0;
            end
        end
    end

    // Activation-stream fork.  Every activation must reach the feedback path;
    // final-layer activations must additionally reach the external output.
    // The two done bits remember a consumer that accepted early.  The HPC sees
    // TREADY only after every required consumer has accepted the current beat,
    // which is the standard "atomic fork" pattern for ready/valid streams.
    always_comb begin
        if (act_packet_is_final) begin
            m_out_axis_tvalid    = act_axis_tvalid & ~out_axis_done_reg;
            m_out_axis_tlast     = act_axis_tlast;
            m_out_axis_tdata     = pk_m_out_axis_tdata;
            act_axis_tready = (out_axis_done_reg | m_out_axis_tready) & (fbk_axis_done_reg | fbk_axis_tready);
        end else begin
            m_out_axis_tvalid    = '0;
            m_out_axis_tlast     = '0;
            m_out_axis_tdata     = '0;
            act_axis_tready = fbk_axis_done_reg | fbk_axis_tready;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n || (act_axis_tvalid & act_axis_tready)) begin
            out_axis_done_reg <= 1'b0;
            fbk_axis_done_reg <= 1'b0;
        end else begin
            if (m_out_axis_tvalid & m_out_axis_tready) begin
                out_axis_done_reg <= 1'b1;
            end
            if (fbk_axis_tvalid & fbk_axis_tready) begin
                fbk_axis_done_reg <= 1'b1;
            end
        end
    end

    // IEU M_AXIS Interface
    assign fbk_axis_tvalid = act_axis_tvalid & ~fbk_axis_done_reg;
    assign fbk_axis_tdata = act_axis_tdata;
    assign fbk_axis_tlast = act_axis_tlast;

    //-----------------------
    //-- Instantiation
    //-----------------------
    ieu # (
        .NUM_PE               (NUM_PE              ),
        .ACT_INT_BW           (ACT_INT_BW          ),
        .ACT_FRA_BW           (ACT_FRA_BW          ),
        .CFG_BW               (CFG_BW              ),
        .CFG_NUM              (CFG_NUM             ),
        .DTH_BW               (DTH_BW              ),
        .NZI_BW               (NZI_BW              ),
        .NZL_FIFO_DEPTH       (NZL_FIFO_DEPTH      ),
        .NUM_LAYER_BW         (NUM_LAYER_BW        ),
        .LAYER_SIZE_BW        (LAYER_SIZE_BW       ),
        .MEM_STATE_DEPTH_BW   (MEM_STATE_DEPTH_BW  )
    ) i_ieu (
        .clk                  (clk                  ),
        .rst_n                (rst_n                ),
        .curr_layer           (ieu_hpc_curr_layer   ),
        .curr_hid_size        (curr_hid_size        ),
        .cfg                  (cfg                  ),               
        .s_inp_axis_tvalid    (s_inp_axis_tvalid    ),
        .s_inp_axis_tready    (s_inp_axis_tready    ),
        .s_inp_axis_tdata     (roll_s_inp_axis_tdata),
        .s_inp_axis_tlast     (s_inp_axis_tlast     ),          
        .s_fbk_axis_tvalid    (fbk_axis_tvalid      ),
        .s_fbk_axis_tready    (fbk_axis_tready      ),
        .s_fbk_axis_tdata     (fbk_axis_tdata       ),
        .s_fbk_axis_tlast     (fbk_axis_tlast       ),               
        .rd_en_hpt            (hpc_ieu_rd_en_hpt    ),
        .dout_hpt             (ieu_hpc_dout_hpt     ),              
        .m_nzv_axis_tvalid    (nzv_axis_tvalid      ),
        .m_nzv_axis_tready    (nzv_axis_tready      ),
        .m_nzv_axis_tdata     (nzv_axis_tdata       ),
        .m_nzv_axis_tlast     (nzv_axis_tlast       ),
        .m_nzi_axis_tvalid    (nzi_axis_tvalid      ),
        .m_nzi_axis_tready    (nzi_axis_tready      ),
        .m_nzi_axis_tdata     (nzi_axis_tdata       ),
        .m_nzi_axis_tlast     (nzi_axis_tlast       )
    );

    hpc # (
        .NUM_PE              (NUM_PE             ),
        .CFG_BW              (CFG_BW             ),
        .CFG_NUM             (CFG_NUM            ),
        .ACT_INT_BW          (ACT_INT_BW         ),
        .ACT_FRA_BW          (ACT_FRA_BW         ),
        .W_INT_BW            (W_INT_BW           ),
        .W_FRA_BW            (W_FRA_BW           ),
        .NUM_LAYER_BW        (NUM_LAYER_BW       ),
        .LAYER_SIZE_BW       (LAYER_SIZE_BW      ),
        .MEM_ACC_DEPTH_BW    (MEM_ACC_DEPTH_BW   ), 
        .MEM_ACC_DEPTH_SL_BW (MEM_ACC_DEPTH_SL_BW)
    ) i_hpc (
        .clk            (clk                   ),
        .rst_n          (rst_n                 ),
        .cfg            (cfg                   ),
        .s_nzv_axis_tvalid (nzv_axis_tvalid    ),
        .s_nzv_axis_tready (nzv_axis_tready    ),
        .s_nzv_axis_tdata  (nzv_axis_tdata     ),
        .s_nzv_axis_tlast  (nzv_axis_tlast     ),
        .s_w_axis_tvalid   (s_w_axis_tvalid    ),
        .s_w_axis_tready   (s_w_axis_tready    ),
        .s_w_axis_tdata    (roll_s_w_axis_tdata),
        .s_w_axis_tlast    (s_w_axis_tlast     ),
        .m_act_axis_tvalid (act_axis_tvalid    ),
        .m_act_axis_tready (act_axis_tready    ),
        .m_act_axis_tdata  (act_axis_tdata     ),
        .m_act_axis_tlast  (act_axis_tlast     ),
        .din_hpt           (ieu_hpc_dout_hpt   ),
        .rd_en_hpt         (hpc_ieu_rd_en_hpt  )
    );

    `ifndef SIM_DEBUG //----------------------------------------------
    s_axi_lite_v1_0 # (
        // Parameters of Axi Slave Bus Interface S_AXI
        C_S_AXI_DATA_WIDTH,
        C_S_AXI_ADDR_WIDTH
    ) i_s_axi_lite	(
        // Users to add ports here
        .reg0  (reg0 ),
        .reg1  (reg1 ),
        .reg2  (reg2 ),
        .reg3  (reg3 ),
        .reg4  (reg4 ),
        .reg5  (reg5 ),
        .reg6  (reg6 ),
        .reg7  (reg7 ),
        .reg8  (reg8 ),
        .reg9  (reg9 ),
        .reg10 (reg10),
        .reg11 (reg11),
        .reg12 (reg12),
        .reg13 (reg13),
        .reg14 (reg14),
        .reg15 (reg15),
        .reg16 (reg16),
        .reg17 (reg17),
        .reg18 (reg18),
        .reg19 (reg19),
        .reg20 (reg20),
        .reg21 (reg21),
        .reg22 (reg22),
        .reg23 (reg23),
        .reg24 (reg24),
        .reg25 (reg25),
        .reg26 (reg26),
        .reg27 (reg27),
        .reg28 (reg28),
        .reg29 (reg29),
        .reg30 (reg30),
        .reg31 (reg31),
        // Ports of Axi Slave Bus Interface S_AXI
        .s_axi_aclk    (clk          ),
        .s_axi_aresetn (rst_n        ),
        .s_axi_awaddr  (s_axi_awaddr ),
        .s_axi_awprot  (s_axi_awprot ),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_axi_wdata  ),
        .s_axi_wstrb   (s_axi_wstrb  ),
        .s_axi_wvalid  (s_axi_wvalid ),
        .s_axi_wready  (s_axi_wready ),
        .s_axi_bresp   (s_axi_bresp  ),
        .s_axi_bvalid  (s_axi_bvalid ),
        .s_axi_bready  (s_axi_bready ),
        .s_axi_araddr  (s_axi_araddr ),
        .s_axi_arprot  (s_axi_arprot ),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata   (s_axi_rdata  ),
        .s_axi_rresp   (s_axi_rresp  ),
        .s_axi_rvalid  (s_axi_rvalid ),
        .s_axi_rready  (s_axi_rready )
    );
    `endif //--------------------------------------------------------

`ifndef SYNTHESIS
    initial begin
        assert (C_S_AXI_DATA_WIDTH == 32 && C_S_AXI_ADDR_WIDTH >= 7)
            else $fatal(1, "AXI-Lite register bank requires 32-bit data and at least 7 address bits");
        assert (NUM_PE >= 2 && is_power_of_two(NUM_PE))
            else $fatal(1, "NUM_PE must be a power of two and at least two");
        assert (MAX_LAYER_SIZE % NUM_PE == 0)
            else $fatal(1, "Maximum layer size must be divisible by NUM_PE");
        assert (CFG_NUM > CFG_IDX_HIDDEN_BASE)
            else $fatal(1, "Configuration register file is too small");
        assert (W_BW == 8)
            else $fatal(1, "Datamover byte addressing currently requires INT8 packed weights");
    end
`endif

endmodule

`default_nettype wire

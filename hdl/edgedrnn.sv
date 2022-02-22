`timescale 1ns/1ps
`include "hdr_macros.v"

module edgedrnn #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 7,
    parameter integer NUM_PE             = 8, // >= 2
    parameter integer ACT_INT_BW         = 8,
    parameter integer ACT_FRA_BW         = 8,
    parameter integer W_INT_BW           = 1,
    parameter integer W_FRA_BW           = 7,
    parameter integer NUM_LAYER_BW       = 2,
    parameter integer LAYER_SIZE_BW      = 10,  
    parameter integer DTH_BW             = 10,
    parameter integer NZI_BW             = 16,
    parameter integer NZL_FIFO_DEPTH     = 32
    )(
    // Ports of Axi-Lite Slave Bus Interface S_AXI
    input  logic                              clk,
    input  logic                              rst_n,
    // Ports of Axi Slave Bus Interface S_AXI
    `ifndef SIM_DEBUG //----------------------------------------------
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  logic [2:0]                        s_axi_awprot,
    input  logic                              s_axi_awvalid,
    output logic                              s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic                              s_axi_wvalid,
    output logic                              s_axi_wready,
    output logic [1:0]                        s_axi_bresp,
    output logic                              s_axi_bvalid,
    input  logic                              s_axi_bready,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  logic [2:0]                        s_axi_arprot,
    input  logic                              s_axi_arvalid,
    output logic                              s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output logic [1:0]                        s_axi_rresp,
    output logic                              s_axi_rvalid,
    input  logic                              s_axi_rready,
    `endif //--------------------------------------------------------
    // S_AXIS Interface (Input)               
    input  logic                                        s_inp_axis_tvalid,
    output logic                                        s_inp_axis_tready,
    input  logic [NUM_PE*(ACT_INT_BW + ACT_FRA_BW)-1:0] s_inp_axis_tdata,
    input  logic                                        s_inp_axis_tlast,
    // S_AXIS Interface (Weight)               
    input  logic                                        s_w_axis_tvalid,
    output logic                                        s_w_axis_tready,
    input  logic [NUM_PE*(W_INT_BW + W_FRA_BW)-1:0]     s_w_axis_tdata,
    input  logic                                        s_w_axis_tlast,
    // M_AXIS Interface (NZIL)            
    output logic                                        m_inst_axis_tvalid,
    input  logic                                        m_inst_axis_tready,
    output logic [80-1:0]                               m_inst_axis_tdata,
    output logic                                        m_inst_axis_tlast,
    // Activation Output
    output logic                                        m_out_axis_tvalid,
    input  logic                                        m_out_axis_tready,
    output logic [NUM_PE*(ACT_INT_BW + ACT_FRA_BW)-1:0] m_out_axis_tdata,
    output logic                                        m_out_axis_tlast
    );

    //-----------------------------------------------------------
    // Useful Functions
    //-----------------------------------------------------------
    // Function returns max of two inputs.
    function automatic int max(int A, int B);
        return (A > B) ? A : B;
    endfunction

    // function called clogb2 that returns an integer which has the 
    // value of the ceiling of the log base 2.                      
    function integer clogb2 (input integer x);              
    begin                                                           
        for(clogb2=0; x>0; clogb2=clogb2+1)                   
            x = x >> 1;                                 
        end                                                           
    endfunction

    // Convert depth to bitwidth
    function integer bw (input integer x);
        integer depth;
        begin
            depth = x - 1;
            for(bw=0; depth>0; bw=bw+1)                   
                depth = depth >> 1;                                 
            end
            depth = max(depth, 1);
    endfunction

    //-----------------------------------------------------------
    // Parameters
    //-----------------------------------------------------------
    localparam ACT_BW                = ACT_INT_BW + ACT_FRA_BW;
    localparam W_BW                  = W_INT_BW + W_FRA_BW;
    localparam ACC_BW                = ACT_BW + W_BW;
    localparam MAX_NUM_LAYER         = 2**NUM_LAYER_BW;
    localparam MAX_LAYER_SIZE        = 2**LAYER_SIZE_BW;
    localparam MAX_WORKSIZE          = MAX_LAYER_SIZE/NUM_PE;
    localparam MEM_STATE_DEPTH       = MAX_LAYER_SIZE/NUM_PE;
    localparam MEM_ACC_DEPTH         = 4*MAX_NUM_LAYER*MAX_LAYER_SIZE/NUM_PE;
    localparam MEM_ACC_DEPTH_SL      = 4*MAX_LAYER_SIZE/NUM_PE;
    localparam CFG_BW                = 32;
    localparam CFG_NUM               = 32;
    // Look-Up Table
    localparam NUM_PE_BW             = clogb2(NUM_PE-1);
    localparam MEM_STATE_DEPTH_BW    = clogb2(MEM_STATE_DEPTH-1);
    localparam MEM_ACC_DEPTH_BW      = NUM_LAYER_BW+LAYER_SIZE_BW+2-NUM_PE_BW;
    localparam MEM_ACC_DEPTH_SL_BW   = LAYER_SIZE_BW+2-NUM_PE_BW;

    //-----------------------
    //-- Logic
    //-----------------------
    genvar g_i;
    // IEU
    logic              nzi_axis_tvalid;
    logic [NZI_BW-1:0] nzi_axis_tdata;
    logic              nzi_axis_tlast;
    logic              nzi_axis_tready;

    logic          [CFG_BW-1:0]       cfg            [CFG_NUM-1:0 ];
    logic                             hpc_ieu_valid;
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
    logic signed [NUM_PE-1:0][ACT_BW-1:0] roll_m_out_axis_tdata;

    // Packed I/O
    logic [NUM_PE*ACT_BW-1:0]   pk_m_out_axis_tdata;

    // HPC M_AXIS Interface
    logic                                 act_axis_tvalid;
    logic                                 act_axis_tready;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] act_axis_tdata ;
    logic                                 act_axis_tlast;

    //-----------------------
    //-- AXI Datamover CMD
    //-----------------------
    logic [15:0] bytes_per_burst;
    logic [31:0] w_addr;
    logic [8:0] w_arlen;
    assign w_arlen = 3*curr_hid_size/NUM_PE;

    always_ff @ (posedge clk) begin
        w_addr <= cfg[CFG_IDX_W_BASEADDR];
        // bytes_per_burst <= w_arlen*(C_M_AXI/_DATA_WIDTH/8);
        bytes_per_burst <= 3*curr_hid_size;
    end
    // Bytes to Transfer
    assign m_inst_axis_tdata[22:0]  = 3*curr_hid_size;
    // Type: INCR
    assign m_inst_axis_tdata[23]    = 1'b1;
    // DSA
    assign m_inst_axis_tdata[29:24] = 6'b0;
    // EOF
    assign m_inst_axis_tdata[30]    = 1'b1;
    // DRR
    assign m_inst_axis_tdata[31]    = 1'b0;
    // Start Address
    assign m_inst_axis_tdata[63:32] = w_addr + nzi_axis_tdata*bytes_per_burst;
    // TAG
    assign m_inst_axis_tdata[67:64] = 4'b0110;
    // RSVD
    assign m_inst_axis_tdata[71:68] = '0;
    // xUSER
    assign m_inst_axis_tdata[75:72] = '1;
    // xCACHE
    assign m_inst_axis_tdata[79:76] = 4'b0010;
    // Flags
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
    generate 
        for (g_i=0; g_i<NUM_PE; g_i=g_i+1) begin: repack_io
            assign roll_s_inp_axis_tdata[g_i][ACT_BW-1:0] = s_inp_axis_tdata[ACT_BW*(g_i+1)-1:ACT_BW*g_i];
            assign roll_s_w_axis_tdata[g_i][W_BW-1:0]   = s_w_axis_tdata[W_BW*(g_i+1)-1:W_BW*g_i];
            assign roll_m_out_axis_tdata[g_i][ACT_BW-1:0] = m_out_axis_tdata[ACT_BW*(g_i+1)-1:ACT_BW*g_i];
            assign pk_m_out_axis_tdata[ACT_BW*(g_i+1)-1:ACT_BW*g_i] = act_axis_tdata[g_i][ACT_BW-1:0];
        end
    endgenerate

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
    parameter CREG_FILE = "../../../../../../hdl/tb/edgedrnn_tb_creg.txt";
    initial begin
        $readmemh(CREG_FILE, cfg);
    end
    `endif //--------------------------------------------------------

    // HPC M_AXIS Interface
    always_comb begin
        if (ieu_hpc_curr_layer == num_layer) begin
            m_out_axis_tvalid    = act_axis_tvalid;
            m_out_axis_tlast     = act_axis_tlast;
            m_out_axis_tdata     = pk_m_out_axis_tdata;
            act_axis_tready = m_out_axis_tready & fbk_axis_tready;
        end else begin
            m_out_axis_tvalid    = '0;
            m_out_axis_tlast     = '0;
            m_out_axis_tdata     = '0;
            act_axis_tready = fbk_axis_tready;
        end
    end

    // IEU M_AXIS Interface
    assign fbk_axis_tvalid = act_axis_tvalid & act_axis_tready;
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

    endmodule

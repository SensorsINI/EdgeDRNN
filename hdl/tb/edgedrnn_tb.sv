//-----------------------------------------------------
// Testbench: DeltaRNN Testbench
// Design Name : deltarnn_tb
// File Name   : deltarnn_tb.sv
//-----------------------------------------------------
`timescale 1ns/1ps
`include "hdr_macros.v"
module edgedrnn_tb;
    //------------------------------------------------------------------
    // Simulator Components
    //------------------------------------------------------------------
    parameter T_CLK_HI     = 2.5;                         // set clock high time
    parameter T_CLK_LO     = 2.5;                         // set clock low time
    parameter T_CLK        = T_CLK_HI + T_CLK_LO; // calculate clock period
    parameter T_INIT       = 200*T_CLK;
    parameter T_APPL_DELAY = 1.5;  // Stimuli application delay
    parameter T_ACQ_DELAY  = 1.5;  // Response aquisition delay
    parameter GOLD_FILE    = "../../../../../../hdl/tb/edgedrnn_tb_gold_rnn.txt";

    //-----------------------------------------------------------
    // Useful Functions
    //-----------------------------------------------------------
    // function called clogb2 that returns an integer which has the 
    // value of the ceiling of the log base 2.                      
    function integer clogb2 (input integer bit_depth);              
    begin                                                           
        for(clogb2=0; bit_depth>0; clogb2=clogb2+1)                   
            bit_depth = bit_depth >> 1;
        end                                                           
    endfunction

    parameter C_S_AXI_DATA_WIDTH       = 32;
    parameter C_S_AXI_ADDR_WIDTH       = 7;
    parameter NUM_PE                   = 8; // > 2
    parameter ACT_INT_BW        = 8;
    parameter ACT_FRA_BW        = 8;
    parameter ACT_BW            = ACT_INT_BW +ACT_FRA_BW;
    parameter W_INT_BW          = 1;
    parameter W_FRA_BW          = 7;
    parameter W_BW              = W_INT_BW+W_FRA_BW;
    parameter NUM_LAYER_BW      = 2;
    parameter LAYER_SIZE_BW     = 10;
    parameter MAX_NUM_LAYER     = 2**NUM_LAYER_BW;
    parameter MAX_LAYER_SIZE    = 2**LAYER_SIZE_BW;
    parameter MAX_WORKSIZE      = MAX_LAYER_SIZE/NUM_PE;
    parameter MEM_STATE_DEPTH   = MAX_LAYER_SIZE/NUM_PE;
    parameter MEM_ACC_DEPTH     = 4*MAX_NUM_LAYER*MAX_LAYER_SIZE/NUM_PE;
    parameter MEM_ACC_DEPTH_SL  = 4*MAX_LAYER_SIZE/NUM_PE;
    parameter CFG_BW            = 32;
    parameter CFG_NUM           = 8;
    parameter DTH_BW            = 10;
    parameter NZI_BW            = 16;
    // Configuration Address Offset
    parameter NUM_LAYER_CFG_OFFSET = 0;
    parameter DTH_CFG_OFFSET       = 1;
    // Hardware Parameters
    parameter NZL_FIFO_DEPTH       = 32;
    // Look-Up Table
    parameter SIGMOID_INPUT_BW     = 12;
    parameter SIGMOID_OUTPUT_BW    = 9;
    parameter TANH_INPUT_BW        = 12;
    parameter TANH_OUTPUT_BW       = 10;
    parameter NUM_TSTEP            = 79;


    //------------------------------------------------
    // Logic
    //------------------------------------------------
    logic                     EndOfSim;
    logic [NUM_PE*ACT_BW-1:0] gold [];
    logic [NUM_PE*ACT_BW-1:0] actual_response, expected_response;
    logic [NUM_PE*ACT_BW-1:0] gold_test;
    longint num_resp, cnt_resp;
    longint timestep;

    // EdgeDRNN Connections
    logic                     s_axi_aclk;
    logic                     s_axi_aresetn;
    logic                     rst_n_input;
    logic                     s_inp_axis_tvalid;
    logic                     s_inp_axis_tready;
    logic [NUM_PE*ACT_BW-1:0] s_inp_axis_tdata;
    logic                     s_inp_axis_tlast;
    logic                     s_w_axis_tvalid;
    logic                     s_w_axis_tready;
    logic [NUM_PE*W_BW-1:0]   s_w_axis_tdata;
    logic                     s_w_axis_tlast;
    logic                     m_inst_axis_tvalid;
    logic                     m_inst_axis_tready;
    logic [80-1:0]            m_inst_axis_tdata;
    logic                     m_inst_axis_tlast;
    logic                     m_out_axis_tvalid;
    logic                     m_out_axis_tready;
    logic [NUM_PE*ACT_BW-1:0] m_out_axis_tdata;
    logic                     m_out_axis_tlast;

    //------------------------------------------------
    // Clock Generator
    //------------------------------------------------
    initial begin
        do begin
            s_axi_aclk = 1'b1; #T_CLK_HI;
            s_axi_aclk = 1'b0; #T_CLK_LO;
        end while (EndOfSim == 1'b0);
    end

    gen_input # (
        .NUM_PE        (NUM_PE),
        .ACT_BW        (ACT_BW),
        .NZI_BW        (NZI_BW)
        ) i_gen_input (
        .s_axi_aclk    (s_axi_aclk),
        .s_axi_aresetn (rst_n_input),
        .m_axis_tvalid (s_inp_axis_tvalid),
        .m_axis_tready (s_inp_axis_tready),
        .m_axis_tdata  (s_inp_axis_tdata),
        .m_axis_tlast  (s_inp_axis_tlast)
    );

    logic                       wmem_sim_fifo_axis_tvalid;
    logic                       wmem_sim_fifo_axis_tready;
    logic [NUM_PE*W_BW-1:0]     wmem_sim_fifo_axis_tdata;
    logic                       wmem_sim_fifo_axis_tlast;

    gen_weight # (
    .NUM_PE        (NUM_PE),
    .W_BW          (W_BW),
    .LAYER_SIZE_BW (LAYER_SIZE_BW)
    ) i_gen_weight (
    .s_axi_aclk    (s_axi_aclk),
    .s_axi_aresetn (s_axi_aresetn),
    .s_axis_tvalid (m_inst_axis_tvalid),
    .s_axis_tready (m_inst_axis_tready),
    .s_axis_tdata  (m_inst_axis_tdata),
    .s_axis_tlast  (m_inst_axis_tlast),        
    .m_axis_tvalid (s_w_axis_tvalid),
    .m_axis_tready (s_w_axis_tready),
    .m_axis_tdata  (s_w_axis_tdata),
    .m_axis_tlast  (s_w_axis_tlast)
    );

    edgedrnn # (
        .C_S_AXI_DATA_WIDTH (C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH (C_S_AXI_ADDR_WIDTH),
        .NUM_PE             (NUM_PE            ),
        .ACT_INT_BW         (ACT_INT_BW        ),
        .ACT_FRA_BW         (ACT_FRA_BW        ),
        .W_INT_BW           (W_INT_BW          ),
        .W_FRA_BW           (W_FRA_BW          ),
        .NUM_LAYER_BW       (NUM_LAYER_BW      ),
        .LAYER_SIZE_BW      (LAYER_SIZE_BW     ),
        .DTH_BW             (DTH_BW            ),
        .NZI_BW             (NZI_BW            ),
        .NZL_FIFO_DEPTH     (NZL_FIFO_DEPTH    )     
    ) i_edgedrnn (
        .clk                (s_axi_aclk        ),
        .rst_n              (s_axi_aresetn     ),
        .s_inp_axis_tvalid  (s_inp_axis_tvalid ),
        .s_inp_axis_tready  (s_inp_axis_tready ),
        .s_inp_axis_tdata   (s_inp_axis_tdata  ),
        .s_inp_axis_tlast   (s_inp_axis_tlast  ),  
        .s_w_axis_tvalid    (s_w_axis_tvalid   ),
        .s_w_axis_tready    (s_w_axis_tready   ),
        .s_w_axis_tdata     (s_w_axis_tdata    ),
        .s_w_axis_tlast     (s_w_axis_tlast    ),  
        .m_inst_axis_tvalid (m_inst_axis_tvalid),
        .m_inst_axis_tready (m_inst_axis_tready),
        .m_inst_axis_tdata  (m_inst_axis_tdata ),
        .m_inst_axis_tlast  (m_inst_axis_tlast ),
        .m_out_axis_tvalid  (m_out_axis_tvalid ),
        .m_out_axis_tready  (m_out_axis_tready ),
        .m_out_axis_tdata   (m_out_axis_tdata  ),
        .m_out_axis_tlast   (m_out_axis_tlast  )
    );
    //-------------------------------------------------------------------------------
    logic signed [ACT_BW-1:0] unpk_m_out_axis_tdata [NUM_PE-1:0];
    genvar g_i;  
    generate 
        for (g_i=0; g_i<NUM_PE; g_i=g_i+1) begin: repack_io
            assign unpk_m_out_axis_tdata[g_i][ACT_BW-1:0] = m_out_axis_tdata[ACT_BW*(g_i+1)-1:ACT_BW*g_i];
        end
    endgenerate

    // Count Total Cycles
    logic [31:0] cycle_counter;
    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            cycle_counter <= '0;
        end else begin
            cycle_counter <= cycle_counter + 1;
        end
    end

    initial begin
        EndOfSim = 0;
        // Read Gold File
        $readmemh(GOLD_FILE, gold);
        num_resp = 0;
        foreach(gold[i]) begin
            num_resp = num_resp + 1;
        end
        gold_test = gold[0];

        // Initialize
        s_axi_aclk    = 1'b1;
        s_axi_aresetn = 1'b0;
        rst_n_input   = 1'b0;
        m_out_axis_tready = 1'b0;

        // Release Reset
        @(posedge s_axi_aclk);
        #T_APPL_DELAY;
        s_axi_aresetn = 1'b1;
        m_out_axis_tready = 1'b1;
        
        // Start to stream in inputs
        # (T_INIT) begin
            @(posedge s_axi_aclk);
            #T_APPL_DELAY;
            rst_n_input   = 1'b1;
        end

        //Compare responses in each cycle
        cnt_resp = 0;
        while (cnt_resp < num_resp) begin
            // Wait for one clock cycle
            @(posedge s_axi_aclk);
            // Calculate current timestep index
            timestep = cnt_resp/(m_inst_axis_tdata[22:0]/3/NUM_PE);
            // Delay response acquistion by the stimuli acquistion delay
            #T_ACQ_DELAY;
            if (m_out_axis_tvalid & m_out_axis_tready) begin
                // Get actual and expected response
                actual_response = m_out_axis_tdata;
                expected_response = gold[cnt_resp];
                // Dispalay results
                $display("Response %5d of %5d | Timestep: %5d | Out: %X | Gold: %X | Time: %d ns", 
                        cnt_resp, num_resp, timestep, actual_response, expected_response, $time);
                // Raise error if response does not match
                assert(actual_response  ==? expected_response) else
                    $error("Mismatch: | Out: %5d | Gold: %5d", actual_response, expected_response);
                cnt_resp = cnt_resp + 1;
            end
        end

        $display("######## Testbench Passed! ########:\n");
        $display("Elapsed Clock Cycles: %d\n", cycle_counter);

        # T_CLK begin
            EndOfSim = 1;
            $finish;
        end
    end

    endmodule

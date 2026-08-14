`timescale 1ns/1ps
`include "hdr_macros.v"
`include "edgedrnn_tb_config.svh"

module edgedrnn_tb;

    localparam realtime T_CLK = 5.0ns;
    localparam realtime T_APPL_DELAY = 1.5ns;
    localparam realtime T_ACQ_DELAY = 1.5ns;
    localparam integer MAX_SIM_CYCLES = 5_000_000;

    localparam integer C_S_AXI_DATA_WIDTH = 32;
    localparam integer C_S_AXI_ADDR_WIDTH = 7;
    localparam integer NUM_PE = 8;
    localparam integer ACT_INT_BW = 8;
    localparam integer ACT_FRA_BW = 8;
    localparam integer ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam integer W_INT_BW = 1;
    localparam integer W_FRA_BW = 7;
    localparam integer W_BW = W_INT_BW + W_FRA_BW;
    localparam integer NUM_LAYER_BW = 2;
    localparam integer LAYER_SIZE_BW = 10;
    localparam integer DTH_BW = 10;
    localparam integer NZI_BW = 16;
    localparam integer NZL_FIFO_DEPTH = 32;

    localparam STIM_FILE = "edgedrnn_tb_stim.txt";
    localparam PARAM_FILE = "edgedrnn_tb_params.txt";
    localparam GOLD_FILE = "edgedrnn_tb_gold_rnn.txt";
    localparam CREG_FILE = "edgedrnn_tb_creg.txt";

    logic s_axi_aclk;
    logic s_axi_aresetn;
    logic rst_n_input;

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
    logic [79:0]              m_inst_axis_tdata;
    logic                     m_inst_axis_tlast;
    logic                     m_out_axis_tvalid;
    logic                     m_out_axis_tready;
    logic [NUM_PE*ACT_BW-1:0] m_out_axis_tdata;
    logic                     m_out_axis_tlast;

    logic [NUM_PE*ACT_BW-1:0] golden [0:`EDGEDRNN_TB_NUM_GOLD-1];
    integer unsigned response_count;
    integer unsigned mismatch_count;
    integer unsigned cycle_count;

    initial begin
        s_axi_aclk = 1'b0;
        forever #(T_CLK / 2.0) s_axi_aclk = ~s_axi_aclk;
    end

    gen_input #(
        .NUM_PE(NUM_PE),
        .ACT_BW(ACT_BW),
        .NUM_STIM(`EDGEDRNN_TB_NUM_STIM),
        .STIM_FILE(STIM_FILE)
    ) i_gen_input (
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(rst_n_input),
        .m_axis_tvalid(s_inp_axis_tvalid),
        .m_axis_tready(s_inp_axis_tready),
        .m_axis_tdata(s_inp_axis_tdata),
        .m_axis_tlast(s_inp_axis_tlast)
    );

    gen_weight #(
        .NUM_PE(NUM_PE),
        .W_BW(W_BW),
        .NUM_PARAMS(`EDGEDRNN_TB_NUM_PARAMS),
        .PARAM_FILE(PARAM_FILE)
    ) i_gen_weight (
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axis_tvalid(m_inst_axis_tvalid),
        .s_axis_tready(m_inst_axis_tready),
        .s_axis_tdata(m_inst_axis_tdata),
        .s_axis_tlast(m_inst_axis_tlast),
        .m_axis_tvalid(s_w_axis_tvalid),
        .m_axis_tready(s_w_axis_tready),
        .m_axis_tdata(s_w_axis_tdata),
        .m_axis_tlast(s_w_axis_tlast)
    );

    edgedrnn #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .NUM_PE(NUM_PE),
        .ACT_INT_BW(ACT_INT_BW),
        .ACT_FRA_BW(ACT_FRA_BW),
        .W_INT_BW(W_INT_BW),
        .W_FRA_BW(W_FRA_BW),
        .NUM_LAYER_BW(NUM_LAYER_BW),
        .LAYER_SIZE_BW(LAYER_SIZE_BW),
        .DTH_BW(DTH_BW),
        .NZI_BW(NZI_BW),
        .NZL_FIFO_DEPTH(NZL_FIFO_DEPTH),
        .CREG_FILE(CREG_FILE)
    ) i_edgedrnn (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .s_inp_axis_tvalid(s_inp_axis_tvalid),
        .s_inp_axis_tready(s_inp_axis_tready),
        .s_inp_axis_tdata(s_inp_axis_tdata),
        .s_inp_axis_tlast(s_inp_axis_tlast),
        .s_w_axis_tvalid(s_w_axis_tvalid),
        .s_w_axis_tready(s_w_axis_tready),
        .s_w_axis_tdata(s_w_axis_tdata),
        .s_w_axis_tlast(s_w_axis_tlast),
        .m_inst_axis_tvalid(m_inst_axis_tvalid),
        .m_inst_axis_tready(m_inst_axis_tready),
        .m_inst_axis_tdata(m_inst_axis_tdata),
        .m_inst_axis_tlast(m_inst_axis_tlast),
        .m_out_axis_tvalid(m_out_axis_tvalid),
        .m_out_axis_tready(m_out_axis_tready),
        .m_out_axis_tdata(m_out_axis_tdata),
        .m_out_axis_tlast(m_out_axis_tlast)
    );

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            cycle_count <= 0;
            m_out_axis_tready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            m_out_axis_tready <= 1'b1;
        end
    end

    initial begin
        $readmemh(GOLD_FILE, golden);
        s_axi_aresetn = 1'b0;
        rst_n_input = 1'b0;
        response_count = 0;
        mismatch_count = 0;

        repeat (4) @(posedge s_axi_aclk);
        #T_APPL_DELAY;
        s_axi_aresetn = 1'b1;

        repeat (200) @(posedge s_axi_aclk);
        #T_APPL_DELAY;
        rst_n_input = 1'b1;

        while ((response_count < `EDGEDRNN_TB_NUM_GOLD) &&
               (cycle_count < MAX_SIM_CYCLES)) begin
            @(posedge s_axi_aclk);
            #T_ACQ_DELAY;
            if (m_out_axis_tvalid && m_out_axis_tready) begin
                if (m_out_axis_tdata !== golden[response_count]) begin
                    mismatch_count = mismatch_count + 1;
                    $error(
                        "Mismatch at response %0d: output=%x golden=%x",
                        response_count,
                        m_out_axis_tdata,
                        golden[response_count]
                    );
                end
                response_count = response_count + 1;
            end
        end

        if (response_count != `EDGEDRNN_TB_NUM_GOLD) begin
            $fatal(
                1,
                "Timed out after %0d cycles with %0d/%0d responses",
                cycle_count,
                response_count,
                `EDGEDRNN_TB_NUM_GOLD
            );
        end
        if (mismatch_count != 0) begin
            $fatal(1, "RTL verification failed with %0d mismatches", mismatch_count);
        end

        $display(
            "RTL_TEST_PASS simulator completed %0d responses in %0d cycles",
            response_count,
            cycle_count
        );
        #(T_CLK);
        $finish;
    end

endmodule

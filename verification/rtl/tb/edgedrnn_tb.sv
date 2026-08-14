`default_nettype none
`include "edgedrnn_tb_config.svh"

/**
 * End-to-end, simulator-independent EdgeDRNN regression.
 *
 * The Python exporter supplies four memories: configuration registers, packed
 * input vectors, packed weights, and golden quantized hidden-state responses.
 * This testbench connects them to the unmodified top-level streaming contract,
 * injects deterministic backpressure, and checks every output bit and TLAST
 * boundary.  The same source list is compiled by xsim and Verilator.
 */
module edgedrnn_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam realtime T_CLK = 5.0ns;
    localparam realtime T_APPL_DELAY = 1.5ns;
    localparam longint unsigned MAX_SIM_CYCLES = 5_000_000;

    localparam int unsigned C_S_AXI_DATA_WIDTH = 32;
    localparam int unsigned C_S_AXI_ADDR_WIDTH = 7;
    localparam int unsigned NUM_PE = 8;
    localparam int unsigned ACT_INT_BW = 8;
    localparam int unsigned ACT_FRA_BW = 8;
    localparam int unsigned ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam int unsigned W_INT_BW = 1;
    localparam int unsigned W_FRA_BW = 7;
    localparam int unsigned W_BW = W_INT_BW + W_FRA_BW;
    localparam int unsigned NUM_LAYER_BW = 2;
    localparam int unsigned LAYER_SIZE_BW = 10;
    localparam int unsigned DTH_BW = 10;
    localparam int unsigned NZI_BW = 16;
    localparam int unsigned NZL_FIFO_DEPTH = 32;
    localparam int unsigned RESPONSE_COUNT_WIDTH =
        $clog2(`EDGEDRNN_TB_NUM_GOLD + 1);

    localparam string STIM_FILE = "edgedrnn_tb_stim.txt";
    localparam string PARAM_FILE = "edgedrnn_tb_params.txt";
    localparam string GOLD_FILE = "edgedrnn_tb_gold_rnn.txt";
    localparam string CREG_FILE = "edgedrnn_tb_creg.txt";

    logic clk;
    logic reset_n;
    logic input_reset_n;

    axis_stream_if #(.DATA_WIDTH(NUM_PE*ACT_BW)) input_axis (
        .clk     (clk),
        .reset_n (input_reset_n)
    );
    axis_stream_if #(.DATA_WIDTH(edgedrnn_pkg::DATAMOVER_CMD_WIDTH)) command_axis (
        .clk     (clk),
        .reset_n (reset_n)
    );
    axis_stream_if #(.DATA_WIDTH(NUM_PE*W_BW)) weight_axis (
        .clk     (clk),
        .reset_n (reset_n)
    );
    axis_stream_if #(.DATA_WIDTH(NUM_PE*ACT_BW)) output_axis (
        .clk     (clk),
        .reset_n (reset_n)
    );

    logic [NUM_PE*ACT_BW-1:0] golden [0:`EDGEDRNN_TB_NUM_GOLD-1];
    typedef logic [RESPONSE_COUNT_WIDTH-1:0] response_count_t;
    response_count_t response_count;
    int unsigned mismatch_count;
    longint unsigned cycle_count;
    logic [15:0] output_ready_lfsr;

    initial begin
        clk = 1'b0;
        forever #(T_CLK / 2.0) clk = ~clk;
    end

    // One packed input word represents one complete timestep in the default
    // eight-input/eight-lane network.  The accelerator uses configured vector
    // lengths rather than input TLAST, so the source preserves legacy TLAST=0.
    axis_file_source #(
        .DATA_WIDTH         (NUM_PE*ACT_BW),
        .NUM_WORDS          (`EDGEDRNN_TB_NUM_STIM),
        .START_DELAY_CYCLES (1_001),
        .LAST_ON_FINAL_WORD (1'b0),
        .MEMORY_FILE        (STIM_FILE)
    ) i_input_source (
        .m_axis (input_axis)
    );

    // Model external DDR + AXI Datamover.  Registered deterministic bubbles
    // exercise the HPC weight-stream backpressure path reproducibly.
    axi_datamover_read_model #(
        .COMMAND_WIDTH  (edgedrnn_pkg::DATAMOVER_CMD_WIDTH),
        .DATA_WIDTH     (NUM_PE*W_BW),
        .NUM_WORDS      (`EDGEDRNN_TB_NUM_PARAMS),
        .INSERT_BUBBLES (1'b1),
        .LFSR_SEED      (16'h5eed),
        .MEMORY_FILE    (PARAM_FILE)
    ) i_weight_memory (
        .command_axis (command_axis),
        .data_axis    (weight_axis)
    );

    edgedrnn #(
        .C_S_AXI_DATA_WIDTH (C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH (C_S_AXI_ADDR_WIDTH),
        .NUM_PE             (NUM_PE),
        .ACT_INT_BW         (ACT_INT_BW),
        .ACT_FRA_BW         (ACT_FRA_BW),
        .W_INT_BW           (W_INT_BW),
        .W_FRA_BW           (W_FRA_BW),
        .NUM_LAYER_BW       (NUM_LAYER_BW),
        .LAYER_SIZE_BW      (LAYER_SIZE_BW),
        .DTH_BW             (DTH_BW),
        .NZI_BW             (NZI_BW),
        .NZL_FIFO_DEPTH     (NZL_FIFO_DEPTH),
        .CREG_FILE          (CREG_FILE)
    ) i_edgedrnn (
        .clk                (clk),
        .rst_n              (reset_n),
        .s_inp_axis_tvalid  (input_axis.tvalid),
        .s_inp_axis_tready  (input_axis.tready),
        .s_inp_axis_tdata   (input_axis.tdata),
        .s_inp_axis_tlast   (input_axis.tlast),
        .s_w_axis_tvalid    (weight_axis.tvalid),
        .s_w_axis_tready    (weight_axis.tready),
        .s_w_axis_tdata     (weight_axis.tdata),
        .s_w_axis_tlast     (weight_axis.tlast),
        .m_inst_axis_tvalid (command_axis.tvalid),
        .m_inst_axis_tready (command_axis.tready),
        .m_inst_axis_tdata  (command_axis.tdata),
        .m_inst_axis_tlast  (command_axis.tlast),
        .m_out_axis_tvalid  (output_axis.tvalid),
        .m_out_axis_tready  (output_axis.tready),
        .m_out_axis_tdata   (output_axis.tdata),
        .m_out_axis_tlast   (output_axis.tlast)
    );

    // Reusable protocol checks cover both testbench producers and DUT
    // producers.  They catch stability errors independently of data equality.
    axis_stream_protocol_checker #(
        .INTERFACE_NAME ("input_axis")
    ) i_input_protocol (
        .axis (input_axis)
    );

    axis_stream_protocol_checker #(
        .INTERFACE_NAME ("command_axis")
    ) i_command_protocol (
        .axis (command_axis)
    );

    axis_stream_protocol_checker #(
        .INTERFACE_NAME ("weight_axis")
    ) i_weight_protocol (
        .axis (weight_axis)
    );

    axis_stream_protocol_checker #(
        .INTERFACE_NAME ("output_axis")
    ) i_output_protocol (
        .axis (output_axis)
    );

    // Deterministic output backpressure also exercises the top-level atomic
    // feedback/output fork.  TREADY is registered, avoiding zero-time loops.
    always_ff @(posedge clk) begin
        if (!reset_n) begin
            output_ready_lfsr <= 16'hcafe;
            output_axis.tready <= 1'b0;
            cycle_count <= '0;
        end else begin
            output_ready_lfsr <= {
                output_ready_lfsr[14:0],
                output_ready_lfsr[15] ^ output_ready_lfsr[13]
                ^ output_ready_lfsr[12] ^ output_ready_lfsr[10]
            };
            output_axis.tready <= output_ready_lfsr[0]
                               || output_ready_lfsr[4];
            cycle_count <= cycle_count + 1'b1;
        end
    end

    function automatic bit expected_last(input response_count_t response_index);
        return ((response_index + 1) % `EDGEDRNN_TB_RESPONSES_PER_TIMESTEP) == 0;
    endfunction

    task automatic report_data_mismatch(input response_count_t response_index);
        $display(
            "RTL_DATA_MISMATCH response=%0d output=%x golden=%x",
            response_index,
            output_axis.tdata,
            golden[response_index]
        );
        for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
            if (output_axis.tdata[pe*ACT_BW +: ACT_BW]
                !== golden[response_index][pe*ACT_BW +: ACT_BW]) begin
                $display(
                    "  lane=%0d output=%0d (0x%04x) golden=%0d (0x%04x)",
                    pe,
                    $signed(output_axis.tdata[pe*ACT_BW +: ACT_BW]),
                    output_axis.tdata[pe*ACT_BW +: ACT_BW],
                    $signed(golden[response_index][pe*ACT_BW +: ACT_BW]),
                    golden[response_index][pe*ACT_BW +: ACT_BW]
                );
            end
        end
    endtask

    // Bit-exact scoreboard.  Sampling at the active clock edge observes the
    // beat that just handshook, before either producer advances via NBA.
    always_ff @(posedge clk) begin
        if (!reset_n) begin
            response_count <= '0;
            mismatch_count <= 0;
        end else if (output_axis.tvalid && output_axis.tready) begin
            if (response_count >= `EDGEDRNN_TB_NUM_GOLD) begin
                $fatal(1, "DUT produced more responses than the golden model");
            end

            if ((output_axis.tdata !== golden[response_count])
                || (output_axis.tlast !== expected_last(response_count))) begin
                mismatch_count <= mismatch_count + 1;
                if (mismatch_count < 10) begin
                    if (output_axis.tdata !== golden[response_count]) begin
                        report_data_mismatch(response_count);
                    end
                    if (output_axis.tlast !== expected_last(response_count)) begin
                        $display(
                            "RTL_TLAST_MISMATCH response=%0d output=%b expected=%b",
                            response_count,
                            output_axis.tlast,
                            expected_last(response_count)
                        );
                    end
                end
            end
            response_count <= response_count + 1'b1;
        end
    end

    initial begin
        assert (`EDGEDRNN_TB_NUM_STIM > 0);
        assert (`EDGEDRNN_TB_NUM_PARAMS > 0);
        assert (`EDGEDRNN_TB_NUM_GOLD > 0);
        assert (`EDGEDRNN_TB_RESPONSES_PER_TIMESTEP > 0);

        $readmemh(GOLD_FILE, golden);
        reset_n = 1'b0;
        input_reset_n = 1'b0;

        repeat (4) @(posedge clk);
        #T_APPL_DELAY;
        reset_n = 1'b1;

        repeat (200) @(posedge clk);
        #T_APPL_DELAY;
        input_reset_n = 1'b1;

        while ((response_count < `EDGEDRNN_TB_NUM_GOLD)
               && (cycle_count < MAX_SIM_CYCLES)) begin
            @(posedge clk);
        end
        #T_APPL_DELAY;

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
            $fatal(1, "RTL verification failed with %0d mismatching responses", mismatch_count);
        end

        $display(
            "RTL_TEST_PASS simulator completed %0d bit-exact responses in %0d cycles",
            response_count,
            cycle_count
        );
        #(T_CLK);
        $finish;
    end

endmodule

`default_nettype wire

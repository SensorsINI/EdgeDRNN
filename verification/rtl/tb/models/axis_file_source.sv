`default_nettype none

/**
 * Registered AXI4-Stream source backed by a hexadecimal memory file.
 *
 * Once TVALID is asserted, TDATA/TLAST stay unchanged until TREADY is sampled
 * high.  This is the key AXI rule that a combinational file reader can easily
 * violate under backpressure.  The initial delay mirrors the legacy testbench
 * and gives the accelerator configuration/state memories time to initialize.
 */
module axis_file_source #(
    parameter int unsigned DATA_WIDTH         = 128,
    parameter int unsigned NUM_WORDS          = 1,
    parameter int unsigned START_DELAY_CYCLES = 1_001,
    parameter bit          LAST_ON_FINAL_WORD = 1'b0,
    parameter string       MEMORY_FILE        = "edgedrnn_tb_stim.txt"
) (
    axis_stream_if.source m_axis
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned INDEX_WIDTH = (NUM_WORDS <= 1) ? 1 : $clog2(NUM_WORDS);
    localparam int unsigned DELAY_WIDTH =
        (START_DELAY_CYCLES <= 1) ? 1 : $clog2(START_DELAY_CYCLES + 1);
    localparam logic [INDEX_WIDTH-1:0] LAST_INDEX = INDEX_WIDTH'(NUM_WORDS - 1);
    localparam logic [DELAY_WIDTH-1:0] LAST_DELAY =
        DELAY_WIDTH'(START_DELAY_CYCLES);

    logic [DATA_WIDTH-1:0] memory [0:NUM_WORDS-1];
    logic [INDEX_WIDTH-1:0] word_index;
    logic [DELAY_WIDTH-1:0] delay_count;
    logic finished;

    always_ff @(posedge m_axis.clk) begin
        if (!m_axis.reset_n) begin
            m_axis.tvalid <= 1'b0;
            m_axis.tdata  <= '0;
            m_axis.tlast  <= 1'b0;
            word_index    <= '0;
            delay_count   <= '0;
            finished      <= 1'b0;
        end else if (!m_axis.tvalid) begin
            if (delay_count < LAST_DELAY) begin
                delay_count <= delay_count + 1'b1;
            end else if (!finished) begin
                m_axis.tvalid <= 1'b1;
                m_axis.tdata  <= memory[word_index];
                m_axis.tlast  <= LAST_ON_FINAL_WORD && (word_index == LAST_INDEX);
            end
        end else if (m_axis.tready) begin
            if (word_index == LAST_INDEX) begin
                m_axis.tvalid <= 1'b0;
                m_axis.tlast  <= 1'b0;
                finished      <= 1'b1;
            end else begin
                word_index    <= word_index + 1'b1;
                m_axis.tdata  <= memory[word_index + 1'b1];
                m_axis.tlast  <= LAST_ON_FINAL_WORD
                               && ((word_index + 1'b1) == LAST_INDEX);
            end
        end
    end

    initial begin
        assert (DATA_WIDTH > 0) else $fatal(1, "DATA_WIDTH must be positive");
        assert (NUM_WORDS > 0) else $fatal(1, "NUM_WORDS must be positive");
        $readmemh(MEMORY_FILE, memory);
    end

endmodule

`default_nettype wire

`default_nettype none

/**
 * Transaction-level read model for the AXI Datamover connection.
 *
 * The DUT emits an 80-bit MM2S command containing a byte address and BTT
 * (bytes to transfer).  This model accepts the command, translates both fields
 * into packed-weight memory indices, and returns a registered AXI stream.
 * Optional deterministic bubbles exercise backpressure while keeping failures
 * reproducible.  Unlike the legacy combinational $urandom source, TVALID and
 * TDATA never change while the DUT holds TREADY low.
 */
module axi_datamover_read_model #(
    parameter int unsigned COMMAND_WIDTH = 80,
    parameter int unsigned DATA_WIDTH    = 64,
    parameter int unsigned NUM_WORDS     = 1,
    parameter bit          INSERT_BUBBLES = 1'b1,
    parameter logic [15:0] LFSR_SEED      = 16'h1,
    parameter string       MEMORY_FILE    = "edgedrnn_tb_params.txt"
) (
    axis_stream_if.sink   command_axis,
    axis_stream_if.source data_axis
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned BYTES_PER_BEAT = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SHIFT = $clog2(BYTES_PER_BEAT);
    localparam int unsigned INDEX_WIDTH = (NUM_WORDS <= 1) ? 1 : $clog2(NUM_WORDS);
    localparam logic [31:0] BYTE_MASK = BYTES_PER_BEAT - 1;
    localparam logic [32:0] NUM_WORDS_FULL = 33'(NUM_WORDS);

    typedef enum logic {MODEL_IDLE, MODEL_READ} model_state_t;
    model_state_t state;

    logic [DATA_WIDTH-1:0] memory [0:NUM_WORDS-1];
    logic [INDEX_WIDTH-1:0] read_index;
    logic [22:0] beats_remaining;
    logic [15:0] lfsr;

    logic [22:0] command_btt_bytes;
    logic [31:0] command_address_bytes;
    logic [31:0] command_word_index_full;
    logic [22:0] command_beat_count;
    logic [32:0] command_end_index_full;
    assign command_btt_bytes = command_axis.tdata[22:0];
    assign command_address_bytes = command_axis.tdata[63:32];
    assign command_word_index_full = command_address_bytes >> BYTE_SHIFT;
    assign command_beat_count = command_btt_bytes >> BYTE_SHIFT;
    assign command_end_index_full = {1'b0, command_word_index_full}
                                  + {{10{1'b0}}, command_beat_count};

    assign command_axis.tready = (state == MODEL_IDLE);

    always_ff @(posedge command_axis.clk) begin
        if (!command_axis.reset_n) begin
            state                <= MODEL_IDLE;
            read_index           <= '0;
            beats_remaining      <= '0;
            lfsr                 <= LFSR_SEED;
            data_axis.tvalid     <= 1'b0;
            data_axis.tdata      <= '0;
            data_axis.tlast      <= 1'b0;
        end else begin
            // x^16 + x^14 + x^13 + x^11 + 1; a non-zero seed never locks up.
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

            case (state)
                MODEL_IDLE: begin
                    data_axis.tvalid <= 1'b0;
                    data_axis.tlast  <= 1'b0;
                    if (command_axis.tvalid) begin
                        assert (command_btt_bytes != 0)
                            else $fatal(1, "Datamover command has zero BTT");
                        assert ((command_btt_bytes & BYTE_MASK[22:0]) == '0)
                            else $fatal(1, "Datamover BTT is not beat-aligned");
                        assert ((command_address_bytes & BYTE_MASK) == '0)
                            else $fatal(1, "Datamover address is not beat-aligned");
                        assert (command_end_index_full <= NUM_WORDS_FULL)
                            else $fatal(1, "Datamover command exceeds parameter memory");

                        read_index      <= command_word_index_full[INDEX_WIDTH-1:0];
                        beats_remaining <= command_beat_count;
                        state           <= MODEL_READ;
                    end
                end

                MODEL_READ: begin
                    if (!data_axis.tvalid) begin
                        if (!INSERT_BUBBLES || lfsr[0] || lfsr[3]) begin
                            data_axis.tvalid <= 1'b1;
                            data_axis.tdata  <= memory[read_index];
                            data_axis.tlast  <= (beats_remaining == 1);
                        end
                    end else if (data_axis.tready) begin
                        data_axis.tvalid <= 1'b0;
                        data_axis.tlast  <= 1'b0;
                        if (beats_remaining == 1) begin
                            beats_remaining <= '0;
                            state           <= MODEL_IDLE;
                        end else begin
                            read_index      <= read_index + 1'b1;
                            beats_remaining <= beats_remaining - 1'b1;
                        end
                    end
                end

                default: begin
                    state            <= MODEL_IDLE;
                    data_axis.tvalid <= 1'b0;
                    data_axis.tlast  <= 1'b0;
                end
            endcase
        end
    end

    initial begin
        assert (COMMAND_WIDTH >= 64)
            else $fatal(1, "COMMAND_WIDTH must contain BTT and address fields");
        assert (DATA_WIDTH > 0 && DATA_WIDTH % 8 == 0)
            else $fatal(1, "DATA_WIDTH must contain a whole number of bytes");
        assert (BYTES_PER_BEAT > 0 && (BYTES_PER_BEAT & (BYTES_PER_BEAT-1)) == 0)
            else $fatal(1, "BYTES_PER_BEAT must be a power of two");
        assert (NUM_WORDS > 0) else $fatal(1, "NUM_WORDS must be positive");
        assert (LFSR_SEED != '0) else $fatal(1, "LFSR_SEED must be non-zero");
        $readmemh(MEMORY_FILE, memory);
    end

endmodule

`default_nettype wire

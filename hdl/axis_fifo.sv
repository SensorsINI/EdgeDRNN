`default_nettype none

/**
 * Single-clock AXI4-Stream FIFO with a registered output.
 *
 * The IEU produces two synchronized streams: a non-zero delta value and its
 * weight-column index.  One instance of this FIFO buffers each stream so that
 * downstream weight fetch and MAC stalls do not stop the delta encoder
 * immediately (paper Fig. 3, D-FIFO and CTRL paths).
 *
 * AXI4-Stream rule for beginners: a transfer occurs only on a rising clock
 * edge where both TVALID and TREADY are high.  Once this module asserts
 * M_AXIS_TVALID, its TDATA/TLAST outputs remain unchanged until accepted.
 * TLAST is stored beside TDATA, so packet boundaries cannot be separated from
 * their payload by backpressure.
 */
module axis_fifo #(
    parameter int unsigned AXIS_BUS_WIDTH = 16,
    // Storage depth is 2**DEPTH_WIDTH words.  The registered output is an
    // additional elastic slot and is not included in the occupancy count.
    parameter int unsigned DEPTH_WIDTH = 10
) (
    input  wire logic                      m_axi_aclk,
    input  wire logic                      m_axi_aresetn,
    input  wire logic [AXIS_BUS_WIDTH-1:0] s_axis_tdata,
    input  wire logic                      s_axis_tvalid,
    input  wire logic                      s_axis_tlast,
    output logic                      s_axis_tready,
    output logic [AXIS_BUS_WIDTH-1:0] m_axis_tdata,
    output logic                      m_axis_tvalid,
    output logic                      m_axis_tlast,
    input  wire logic                      m_axis_tready
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned DEPTH = 1 << DEPTH_WIDTH;
    localparam logic [DEPTH_WIDTH:0] DEPTH_COUNT =
        {1'b1, {DEPTH_WIDTH{1'b0}}};
    typedef logic [AXIS_BUS_WIDTH:0] fifo_word_t;

    fifo_word_t memory [DEPTH-1:0];
    fifo_word_t fifo_input;
    fifo_word_t fifo_output;

    logic [DEPTH_WIDTH-1:0] read_pointer;
    logic [DEPTH_WIDTH-1:0] write_pointer;
    logic [DEPTH_WIDTH:0]   occupancy;
    logic                   empty;
    logic                   full;
    logic                   write_enable;
    logic                   read_enable;
    logic                   next_output_valid;

    // These equations intentionally mirror the original registered-output
    // FIFO.  In particular, occupancy counts words still in RAM; a word is
    // removed as soon as it is prefetched into the output register.
    always_comb begin
        empty = (occupancy == '0);
        full  = (occupancy == DEPTH_COUNT);

        s_axis_tready = !full;
        write_enable  = s_axis_tvalid && s_axis_tready;
        read_enable   = (m_axis_tready && m_axis_tvalid && !empty)
                      || (!m_axis_tvalid && !empty);
        fifo_input    = {s_axis_tlast, s_axis_tdata};
    end

    assign fifo_output = memory[read_pointer];
    assign next_output_valid = (m_axis_tready && !empty)
                             || (!m_axis_tvalid && !empty);

    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            write_pointer <= '0;
        end else if (write_enable) begin
            write_pointer <= write_pointer + 1'b1;
        end
    end

    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            read_pointer <= '0;
        end else if (read_enable) begin
            read_pointer <= read_pointer + 1'b1;
        end
    end

    // RAM contents are deliberately not reset, preserving BRAM inference.
    always_ff @(posedge m_axi_aclk) begin
        if (write_enable) begin
            memory[write_pointer] <= fifo_input;
        end
    end

    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            m_axis_tdata <= '0;
            m_axis_tlast <= 1'b0;
        end else if ((m_axis_tvalid && m_axis_tready)
                  || (!m_axis_tvalid && !empty)) begin
            m_axis_tdata <= fifo_output[AXIS_BUS_WIDTH-1:0];
            m_axis_tlast <= fifo_output[AXIS_BUS_WIDTH] && next_output_valid;
        end
    end

    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            m_axis_tvalid <= 1'b0;
        end else if (m_axis_tready || (!m_axis_tvalid && !empty)) begin
            m_axis_tvalid <= next_output_valid;
        end
    end

    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            occupancy <= '0;
        end else if (write_enable && !read_enable && !full) begin
            occupancy <= occupancy + 1'b1;
        end else if (!write_enable && read_enable && !empty) begin
            occupancy <= occupancy - 1'b1;
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (AXIS_BUS_WIDTH > 0)
            else $fatal(1, "AXIS_BUS_WIDTH must be positive");
        assert (DEPTH_WIDTH > 0)
            else $fatal(1, "DEPTH_WIDTH must be positive");
    end
`endif

endmodule

`default_nettype wire

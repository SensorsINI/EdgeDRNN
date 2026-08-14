`default_nettype none

/**
 * Banked state storage used by the Input Encoding Unit (IEU).
 *
 * DeltaGRU needs more history than a conventional GRU (paper Eq. (2)):
 *
 *   current banks  - the latest h_t produced for each layer; these values are
 *                    also read by the HPE while calculating the next h_t.
 *   previous banks - the thresholded reference vectors x_hat and h_hat.  A
 *                    reference element is updated only when its delta reaches
 *                    the configured threshold.
 *
 * Both memories are striped over NUM_PE BRAM banks.  A packed AXI beat maps
 * lane pe to bank pe, so a complete group of NUM_PE state elements can be
 * read in parallel.  The previous-state memory has one extra address bit
 * because the IEU stores input-reference and hidden-reference regions.
 */
module mem_state #(
    parameter int unsigned NUM_PE             = 16,
    parameter int unsigned ACT_INT_BW         = 8,
    parameter int unsigned ACT_FRA_BW         = 8,
    parameter int unsigned NUM_LAYER_BW       = 2,
    parameter int unsigned MEM_STATE_DEPTH_BW = 5
) (
    input  wire logic                                         clk,
`ifdef SIM_DEBUG
    input  wire logic                                         rst_n,
`endif
    // Latest/current state banks.
    input  wire logic                                         curr_wr_en,
    input  wire logic                                         curr_rd_en,
    input  wire logic unsigned [NUM_LAYER_BW-1:0]             curr_l_wr_addr,
    input  wire logic unsigned [NUM_LAYER_BW-1:0]             curr_l_rd_addr,
    input  wire logic unsigned [MEM_STATE_DEPTH_BW-1:0]       curr_wr_addr,
    input  wire logic unsigned [MEM_STATE_DEPTH_BW-1:0]       curr_rd_addr,
    input  wire logic signed [NUM_PE-1:0][ACT_INT_BW+ACT_FRA_BW-1:0] curr_din,
    output logic signed [NUM_PE-1:0][ACT_INT_BW+ACT_FRA_BW-1:0] curr_dout,
    // Thresholded-reference state banks.
    input  wire logic          [NUM_PE-1:0]                   prev_wr_en,
    input  wire logic                                         prev_rd_en,
    input  wire logic unsigned [NUM_LAYER_BW-1:0]             prev_l_wr_addr,
    input  wire logic unsigned [NUM_LAYER_BW-1:0]             prev_l_rd_addr,
    input  wire logic unsigned [MEM_STATE_DEPTH_BW:0]         prev_wr_addr,
    input  wire logic unsigned [MEM_STATE_DEPTH_BW:0]         prev_rd_addr,
    input  wire logic signed [NUM_PE-1:0][ACT_INT_BW+ACT_FRA_BW-1:0] prev_din,
    output logic signed [NUM_PE-1:0][ACT_INT_BW+ACT_FRA_BW-1:0] prev_dout
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam int unsigned CURR_ADDR_BW = NUM_LAYER_BW + MEM_STATE_DEPTH_BW;
    localparam int unsigned PREV_ADDR_BW = CURR_ADDR_BW + 1;

    logic [CURR_ADDR_BW-1:0] current_bank_wr_addr;
    logic [CURR_ADDR_BW-1:0] current_bank_rd_addr;
    logic [PREV_ADDR_BW-1:0] previous_bank_wr_addr;
    logic [PREV_ADDR_BW-1:0] previous_bank_rd_addr;

    // Layer bits are the most significant address bits, making every layer a
    // contiguous memory region and simplifying BRAM inference.
    assign current_bank_wr_addr  = {curr_l_wr_addr, curr_wr_addr};
    assign current_bank_rd_addr  = {curr_l_rd_addr, curr_rd_addr};
    assign previous_bank_wr_addr = {prev_l_wr_addr, prev_wr_addr};
    assign previous_bank_rd_addr = {prev_l_rd_addr, prev_rd_addr};

    for (genvar pe = 0; pe < NUM_PE; pe++) begin : g_state_bank
        bram_sdp #(
            .DATA_BIT_WIDTH  (ACT_BW),
            .DEPTH_BIT_WIDTH (CURR_ADDR_BW)
        ) i_current_state (
            .clk     (clk),
`ifdef SIM_DEBUG
            .rst_n   (rst_n),
`endif
            .cs      (1'b1),
            .wr_en   (curr_wr_en),
            .rd_en   (curr_rd_en),
            .addr_wr (current_bank_wr_addr),
            .addr_rd (current_bank_rd_addr),
            .din     (curr_din[pe]),
            .dout    (curr_dout[pe])
        );

        bram_sdp #(
            .DATA_BIT_WIDTH  (ACT_BW),
            .DEPTH_BIT_WIDTH (PREV_ADDR_BW)
        ) i_previous_state (
            .clk     (clk),
`ifdef SIM_DEBUG
            .rst_n   (rst_n),
`endif
            .cs      (1'b1),
            .wr_en   (prev_wr_en[pe]),
            .rd_en   (prev_rd_en),
            .addr_wr (previous_bank_wr_addr),
            .addr_rd (previous_bank_rd_addr),
            .din     (prev_din[pe]),
            .dout    (prev_dout[pe])
        );
    end

`ifndef SYNTHESIS
    initial begin
        assert (NUM_PE > 0) else $fatal(1, "NUM_PE must be positive");
        assert (ACT_BW > 0) else $fatal(1, "Activation width must be positive");
    end
`endif

endmodule

`default_nettype wire

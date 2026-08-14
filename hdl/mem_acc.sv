`default_nettype none

/**
 * Banked accumulation memory for DeltaGRU delta memories.
 *
 * The paper calls this block "ACC Mem" (Figs. 5-7).  Each GRU layer owns four
 * H-element regions: M_r, M_u, M_xc, and M_hc from Eq. (3).  A sparse matrix
 * column update reads an old partial sum, the HPE adds delta*weight, and the
 * result is written back.  Activation generation later reads the same four
 * regions to calculate r, u, c, and h.
 *
 * There is one BRAM bank per processing element.  Every bank uses the same
 * address, allowing NUM_PE neuron rows to be updated in parallel by a single
 * weight beat.  addr_layer selects a layer; addr_rd/addr_wr select a location
 * within that layer.
 */
module mem_acc #(
    parameter int unsigned NUM_PE              = 16,
    parameter int unsigned ACC_BW              = 32,
    parameter int unsigned NUM_LAYER_BW        = 2,
    parameter int unsigned MEM_ACC_DEPTH_BW    = 9,
    parameter int unsigned MEM_ACC_DEPTH_SL_BW = 7
) (
    input  wire logic                                     clk,
`ifdef SIM_DEBUG
    input  wire logic                                     rst_n,
`endif
    input  wire logic                                     wr_en,
    input  wire logic                                     rd_en,
    input  wire logic unsigned [NUM_LAYER_BW-1:0]         addr_layer,
    input  wire logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0]  addr_wr,
    input  wire logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0]  addr_rd,
    input  wire logic signed   [NUM_PE-1:0][ACC_BW-1:0]   din,
    output logic signed   [NUM_PE-1:0][ACC_BW-1:0]   dout
);
    timeunit 1ns;
    timeprecision 1ps;

    // Concatenation makes each layer a contiguous address region.
    logic [MEM_ACC_DEPTH_BW-1:0] bank_addr_wr;
    logic [MEM_ACC_DEPTH_BW-1:0] bank_addr_rd;
    assign bank_addr_wr = {addr_layer, addr_wr};
    assign bank_addr_rd = {addr_layer, addr_rd};

    for (genvar pe = 0; pe < NUM_PE; pe++) begin : g_accumulator_bank
        bram_sdp #(
            .DATA_BIT_WIDTH  (ACC_BW),
            .DEPTH_BIT_WIDTH (MEM_ACC_DEPTH_BW)
        ) i_bram (
            .clk     (clk),
`ifdef SIM_DEBUG
            .rst_n   (rst_n),
`endif
            .cs      (1'b1),
            .wr_en   (wr_en),
            .rd_en   (rd_en),
            .addr_wr (bank_addr_wr),
            .addr_rd (bank_addr_rd),
            .din     (din[pe]),
            .dout    (dout[pe])
        );
    end

`ifndef SYNTHESIS
    initial begin
        assert (NUM_PE > 0) else $fatal(1, "NUM_PE must be positive");
        assert (MEM_ACC_DEPTH_BW == NUM_LAYER_BW + MEM_ACC_DEPTH_SL_BW)
            else $fatal(1, "Accumulation-memory address widths are inconsistent");
    end
`endif

endmodule

`default_nettype wire

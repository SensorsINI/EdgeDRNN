`default_nettype none

/**
 * Inference-friendly simple dual-port block RAM.
 *
 * One logical port writes and one logical port reads on the same clock.  Reads
 * are synchronous: asserting rd_en at cycle N updates dout after that rising
 * edge.  When the same address is read and written together, nonblocking
 * assignment semantics produce the old value ("read first" behavior).
 *
 * The optional reset exists only in SIM_DEBUG builds.  Clearing every memory
 * word is useful for deterministic RTL verification, but must be omitted for
 * synthesis so Vivado can infer an actual BRAM instead of resettable flops.
 */
module bram_sdp #(
    parameter int unsigned DATA_BIT_WIDTH  = 32,
    parameter int unsigned DEPTH_BIT_WIDTH = 9
) (
    input  wire logic                                 clk,
`ifdef SIM_DEBUG
    input  wire logic                                 rst_n,
`endif
    input  wire logic                                 cs,
    input  wire logic                                 wr_en,
    input  wire logic                                 rd_en,
    input  wire logic unsigned [DEPTH_BIT_WIDTH-1:0] addr_wr,
    input  wire logic unsigned [DEPTH_BIT_WIDTH-1:0] addr_rd,
    input  wire logic signed   [DATA_BIT_WIDTH-1:0]  din,
    output logic signed   [DATA_BIT_WIDTH-1:0]  dout
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned DEPTH = 1 << DEPTH_BIT_WIDTH;
    logic signed [DATA_BIT_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge clk) begin
`ifdef SIM_DEBUG
        if (!rst_n) begin
            for (int unsigned address = 0; address < DEPTH; address++) begin
                memory[address] <= '0;
            end
        end else begin
`endif
            if (cs && wr_en) begin
                memory[addr_wr] <= din;
            end
            if (cs && rd_en) begin
                dout <= memory[addr_rd];
            end
`ifdef SIM_DEBUG
        end
`endif
    end

`ifndef SYNTHESIS
    initial begin
        assert (DATA_BIT_WIDTH > 0)
            else $fatal(1, "DATA_BIT_WIDTH must be positive");
        assert (DEPTH_BIT_WIDTH > 0)
            else $fatal(1, "DEPTH_BIT_WIDTH must be positive");
    end
`endif

endmodule

`default_nettype wire

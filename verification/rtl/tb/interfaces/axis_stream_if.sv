`default_nettype none

/**
 * Minimal AXI4-Stream interface shared by the portable RTL testbench.
 *
 * Bundling the handshake signals prevents accidental cross-connection as the
 * testbench grows.  Modports document signal ownership without tying the
 * reusable models to an AMD or Verilator-specific verification library.
 */
interface axis_stream_if #(
    parameter int unsigned DATA_WIDTH = 32
) (
    input wire logic clk,
    input wire logic reset_n
);
    timeunit 1ns;
    timeprecision 1ps;

    logic                  tvalid;
    logic                  tready;
    logic [DATA_WIDTH-1:0] tdata;
    logic                  tlast;

    modport source (
        input  clk, reset_n, tready,
        output tvalid, tdata, tlast
    );

    modport sink (
        input  clk, reset_n, tvalid, tdata, tlast,
        output tready
    );

    modport monitor (
        input clk, reset_n, tvalid, tready, tdata, tlast
    );

endinterface

`default_nettype wire

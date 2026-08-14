`ifndef HDR_MACROS_V
`define HDR_MACROS_V

// Legacy compatibility header.
//
// Modern EdgeDRNN RTL imports shared constants from edgedrnn_pkg.sv.  Keep
// this include guard because older Vivado block designs may still list this
// header, but do not place parameters in compilation-unit scope here.
//
// Simulation mode is selected by the simulator command line (`-DSIM_DEBUG`).
// Defining it in a source file is unsafe because the same RTL is synthesized.

`endif

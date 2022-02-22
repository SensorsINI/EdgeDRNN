/******************************
 * bram_sdp.sv
 ******************************/
/* FUNCTION:
 * Block RAM in SDP mode
 *
 * VERSION DESCRIPTION:
 * V1.0
 */
`timescale 1ns/1ps
`include "hdr_macros.v"
//`define SIM_DEBUG

module bram_sdp #(
   DATA_BIT_WIDTH = 32,
   DEPTH_BIT_WIDTH = 9
   )(
   input  logic                                clk,
`ifdef SIM_DEBUG //----------------------------------------------
   input  logic                                rst_n,
`endif //--------------------------------------------------------
   input  logic                                cs,
   input  logic                                wr_en,
   input  logic                                rd_en,
   input  logic unsigned [DEPTH_BIT_WIDTH-1:0] addr_wr,
   input  logic unsigned [DEPTH_BIT_WIDTH-1:0] addr_rd,
   input  logic signed   [DATA_BIT_WIDTH-1:0]  din,
   output logic signed   [DATA_BIT_WIDTH-1:0]  dout
   );
   
   // Memory
   logic signed [DATA_BIT_WIDTH-1:0] mem_data [2**DEPTH_BIT_WIDTH-1:0];
   
   // Read & Write
   always_ff @ (posedge clk) begin
`ifdef SIM_DEBUG //----------------------------------------------
      if (!rst_n) begin
         for (int unsigned i=0; i<2**DEPTH_BIT_WIDTH; i=i+1) begin
           mem_data[i] <= '0;
         end
      end else begin
`endif //--------------------------------------------------------
         if (cs & wr_en) begin
            mem_data[addr_wr] <= din;
         end
      
         if (cs & rd_en) begin
            dout <= mem_data[addr_rd];
         end
`ifdef SIM_DEBUG //----------------------------------------------
      end
`endif //--------------------------------------------------------
   end
      
endmodule

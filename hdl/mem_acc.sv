/******************************
 * mem_acc.sv
 ******************************/
/* FUNCTION:
 * Accumulation Memory Block
 *
 * VERSION DESCRIPTION:
 * V1.0
 */
`include "hdr_macros.v"
//`define SIM_DEBUG

module mem_acc #(
   NUM_PE              = 16,
   ACC_BW              = 32,
   NUM_LAYER_BW        = 2,    
   MEM_ACC_DEPTH_BW    = 9,  
   MEM_ACC_DEPTH_SL_BW = 7
   )(
   input  logic                                    clk,
`ifdef SIM_DEBUG //----------------------------------------------
   input  logic                                    rst_n,
`endif //--------------------------------------------------------
   input  logic                                    wr_en,
   input  logic                                    rd_en,
   input  logic unsigned [NUM_LAYER_BW-1:0]        addr_layer,
   input  logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0] addr_wr,
   input  logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0] addr_rd,
   input  logic signed   [NUM_PE-1:0][ACC_BW-1:0]  din,
   output logic signed   [NUM_PE-1:0][ACC_BW-1:0]  dout 
   );
   
   // BRAM Instantiation
   genvar g_i;
   generate
   for (g_i = 0; g_i < NUM_PE; g_i = g_i + 1) begin: gen_bram
      bram_sdp #(
         ACC_BW,
         MEM_ACC_DEPTH_BW
      ) i_bram (
         .clk     (clk),
      `ifdef SIM_DEBUG //----------------------------------------------
         .rst_n   (rst_n),
      `endif //--------------------------------------------------------
         .cs      (1'b1),
         .wr_en   (wr_en),
         .rd_en   (rd_en),
         .addr_wr ({addr_layer, addr_wr}),
         .addr_rd ({addr_layer, addr_rd}),
         .din     (din[g_i]),
         .dout    (dout[g_i])
      );
   end
   endgenerate
   

endmodule

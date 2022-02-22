/******************************
 * mem_state.sv
 ******************************/
/* FUNCTION:
 * Memory block to store previous states
 *
 * VERSION DESCRIPTION:
 * V1.0
 */
`include "hdr_macros.v"

module mem_state # (
   NUM_PE             = 16,
   ACT_INT_BW         = 8,
   ACT_FRA_BW         = 8,
   NUM_LAYER_BW       = 2,
   MEM_STATE_DEPTH_BW = 5
   )(
   input  logic                                        clk,
`ifdef SIM_DEBUG //----------------------------------------------
   input  logic                                        rst_n,
`endif //--------------------------------------------------------
   // Current State Banks Interface
   input  logic                                                    curr_wr_en,
   input  logic                                                    curr_rd_en,
   input  logic unsigned [NUM_LAYER_BW-1:0]                        curr_l_wr_addr,
   input  logic unsigned [NUM_LAYER_BW-1:0]                        curr_l_rd_addr,
   input  logic unsigned [MEM_STATE_DEPTH_BW-1:0]                  curr_wr_addr,
   input  logic unsigned [MEM_STATE_DEPTH_BW-1:0]                  curr_rd_addr,
   input  logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] curr_din,
   output logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] curr_dout,
   // Previous State Banks Interface
   input  logic          [NUM_PE-1:0]                              prev_wr_en,
   input  logic                                                    prev_rd_en,
   input  logic unsigned [NUM_LAYER_BW-1:0]                        prev_l_wr_addr,
   input  logic unsigned [NUM_LAYER_BW-1:0]                        prev_l_rd_addr,
   input  logic unsigned [MEM_STATE_DEPTH_BW:0]                    prev_wr_addr,
   input  logic unsigned [MEM_STATE_DEPTH_BW:0]                    prev_rd_addr,
   input  logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] prev_din    ,
   output logic signed   [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0] prev_dout   
   );
   
   //-----------------------------------------------------------
   // Useful Functions
   //-----------------------------------------------------------
   
   // function called clogb2 that returns an integer which has the 
	// value of the ceiling of the log base 2.                      
	// function called clogb2 that returns an integer which has the 
	// value of the ceiling of the log base 2.                      
	function automatic integer clogb2;
      input integer bit_depth;
      integer temp_bit_depth;
   begin
      temp_bit_depth = bit_depth;
      for(clogb2=0; temp_bit_depth>0; clogb2=clogb2+1)                   
         temp_bit_depth = temp_bit_depth >> 1;                                 
      end                                                           
	endfunction

   //-----------------------------------------------------------
   // Local Parameters
   //-----------------------------------------------------------
   localparam ACT_BW = ACT_INT_BW + ACT_FRA_BW;

   //-----------------------------------------------------------
   // Logic
   //-----------------------------------------------------------

   // Internal Signals
   logic unsigned [NUM_LAYER_BW+MEM_STATE_DEPTH_BW-1:0] mem_curr_wr_addr;
   logic unsigned [NUM_LAYER_BW+MEM_STATE_DEPTH_BW-1:0] mem_curr_rd_addr;
   logic unsigned [NUM_LAYER_BW+MEM_STATE_DEPTH_BW:0]   mem_prev_wr_addr;
   logic unsigned [NUM_LAYER_BW+MEM_STATE_DEPTH_BW:0]   mem_prev_rd_addr;
   
   logic signed   [NUM_PE-1:0][ACT_BW-1:0]              mem_curr_din  ;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0]              mem_curr_dout ;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0]              mem_prev_din  ;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0]              mem_prev_dout ;
   
   // Assign I/O Ports
   assign mem_curr_din = curr_din;
   assign mem_prev_din = prev_din;
   assign curr_dout = mem_curr_dout;
   assign prev_dout = mem_prev_dout;
   
   // Memory Address Encoding (Port A for curr; Port B for prev)
   assign mem_curr_wr_addr = {curr_l_wr_addr, curr_wr_addr};
   assign mem_curr_rd_addr = {curr_l_rd_addr, curr_rd_addr};
   assign mem_prev_wr_addr = {prev_l_wr_addr, prev_wr_addr};
   assign mem_prev_rd_addr = {prev_l_rd_addr, prev_rd_addr};
   
   // Instantiate Left BRAMs in TDP Mode
   genvar g_i;
   generate
   for (g_i = 0; g_i < NUM_PE; g_i = g_i + 1) begin: gen_mem_curr
      bram_sdp #(
         ACT_BW,
         NUM_LAYER_BW + MEM_STATE_DEPTH_BW
      ) i_mem_curr (
         .clk     (clk),
   `ifdef SIM_DEBUG //----------------------------------------------
         .rst_n   (rst_n),
   `endif //--------------------------------------------------------
         .cs      (1'b1),
         .wr_en   (curr_wr_en),
         .rd_en   (curr_rd_en),
         .addr_wr (mem_curr_wr_addr),
         .addr_rd (mem_curr_rd_addr),
         .din     (mem_curr_din [g_i]),
         .dout    (mem_curr_dout[g_i])
      );
   end
   endgenerate
   
   // Instantiate Right BRAMs in TDP Mode on
   generate
   for (g_i = 0; g_i < NUM_PE; g_i = g_i + 1) begin: gen_mem_prev
      bram_sdp #(
        ACT_BW,
        NUM_LAYER_BW + MEM_STATE_DEPTH_BW + 1
      ) i_mem_prev (
         .clk     (clk),
   `ifdef SIM_DEBUG //----------------------------------------------
         .rst_n   (rst_n),
   `endif //--------------------------------------------------------
         .cs      (1'b1),
         .wr_en   (prev_wr_en[g_i]),
         .rd_en   (prev_rd_en),
         .addr_wr (mem_prev_wr_addr),
         .addr_rd (mem_prev_rd_addr),
         .din     (mem_prev_din [g_i]),
         .dout    (mem_prev_dout[g_i])
      );
   end
   endgenerate
   
endmodule

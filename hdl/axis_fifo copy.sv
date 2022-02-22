`timescale 1ns/1ps
`include "hdr_macros.v"

module axis_fifo # (
      parameter AXIS_BUS_WIDTH = 16,
      parameter DEPTH_WIDTH = 10
   )(
      input  logic                      m_axi_aclk,
      input  logic                      m_axi_aresetn,
      input  logic [AXIS_BUS_WIDTH-1:0] s_axis_tdata,
      input  logic                      s_axis_tvalid,
      input  logic                      s_axis_tlast,
      output logic                      s_axis_tready,
      output logic [AXIS_BUS_WIDTH-1:0] m_axis_tdata,
      output logic                      m_axis_tvalid,
      output logic                      m_axis_tlast,
      input  logic                      m_axis_tready
   );
   
   // Logics
   logic empty, full;
   logic re,we;
   logic [AXIS_BUS_WIDTH:0] fifo_dout;
   logic [AXIS_BUS_WIDTH:0] fifo_din;

   always_comb begin
      m_axis_tvalid = !empty;
      s_axis_tready = !full;
      we = s_axis_tvalid & !full;
      re = m_axis_tready & m_axis_tvalid;
      fifo_din[AXIS_BUS_WIDTH] = s_axis_tlast;
      fifo_din[AXIS_BUS_WIDTH-1:0] = s_axis_tdata;
      m_axis_tlast = fifo_dout[AXIS_BUS_WIDTH] & !empty;
      m_axis_tdata = fifo_dout[AXIS_BUS_WIDTH-1:0];
   end
   
   //Pointers
   logic [DEPTH_WIDTH-1:0] read_pointer;
   logic [DEPTH_WIDTH-1:0] write_pointer;
   logic [DEPTH_WIDTH:0]   status_pointer;

   //Flags
   always_comb begin
      full  = (status_pointer == 2**DEPTH_WIDTH);
      empty = (status_pointer == 0);
   end
   
   //Memorys
   logic [AXIS_BUS_WIDTH:0] mem_data [2**DEPTH_WIDTH-1:0];
   
   //Write Pointer
   always_ff @ (posedge m_axi_aclk) begin
      if (!m_axi_aresetn) begin
         write_pointer <= '0;
      end else if (we) begin
         mem_data[write_pointer] <= fifo_din;
         write_pointer <= write_pointer + 1;
      end
   end
   
   //Read Pointer
   always_ff @ (posedge m_axi_aclk) begin
      if (!m_axi_aresetn) begin
         read_pointer <= '0;
      end else if (re) begin
         read_pointer <= read_pointer + 1;
      end
   end
   
   // Output Data
   assign fifo_dout = mem_data [read_pointer];
    
   //Status
   always_ff @ (posedge m_axi_aclk) begin
      if (!m_axi_aresetn) begin
         status_pointer <= '0;
      end else if (we && !re && !full) begin   //Only write
         status_pointer <= status_pointer + 1;
      end else if (!we && re && !empty) begin  //Only read
         status_pointer <= status_pointer - 1;
      end
   end

endmodule

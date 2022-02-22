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
    logic re, we;
    logic [AXIS_BUS_WIDTH:0] fifo_dout;
    logic [AXIS_BUS_WIDTH:0] fifo_din;
    logic n_m_axis_tvalid;

    always_comb begin
        s_axis_tready = ~full;
        we = s_axis_tvalid & s_axis_tready;
        re = (m_axis_tready & m_axis_tvalid & ~empty) | (~m_axis_tvalid & ~empty);
        fifo_din[AXIS_BUS_WIDTH] = s_axis_tlast;
        fifo_din[AXIS_BUS_WIDTH-1:0] = s_axis_tdata;
    end

    //Pointers
    logic [DEPTH_WIDTH-1:0] rptr;
    logic [DEPTH_WIDTH-1:0] wptr;
    logic [DEPTH_WIDTH:0]   sptr;

    // Full Flags
    always_comb begin
        if (sptr == $unsigned(2**DEPTH_WIDTH)) begin
            full = 1'b1;
        end else begin
            full = 1'b0;
        end
    end

    // Empty Flags
    always_comb begin
        if (sptr == 0) begin
            empty = 1'b1;
        end else begin
            empty = 1'b0;
        end
    end

    //Memorys
    logic [AXIS_BUS_WIDTH:0] mem_data [2**DEPTH_WIDTH-1:0];

    //Write Pointer
    always_ff @ (posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            wptr <= '0;
        end else if (we) begin
            wptr <= wptr + 1;
        end
    end

    //Read Pointer
    always_ff @ (posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            rptr <= '0;
        end else if (re) begin // && !empty
            rptr <= rptr + 1;
        end
    end

    // Memory
    always_ff @ (posedge m_axi_aclk) begin
        if (we) begin
            mem_data[wptr] <= fifo_din;
        end
    end

    // Output Data
    assign fifo_dout = mem_data [rptr];

    // Output
    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            m_axis_tdata <= '0;
            m_axis_tlast <= '0;
        end else if ((m_axis_tvalid & m_axis_tready) || (~m_axis_tvalid & ~empty)) begin
            m_axis_tdata <= fifo_dout[AXIS_BUS_WIDTH-1:0];
            m_axis_tlast <= fifo_dout[AXIS_BUS_WIDTH] & n_m_axis_tvalid;
        end
    end

    // Valid
    assign n_m_axis_tvalid = (m_axis_tready & ~empty) | (~m_axis_tvalid & ~empty);
    always_ff @(posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            m_axis_tvalid <= '0;
        end else if (m_axis_tready || (~m_axis_tvalid & ~empty)) begin
            m_axis_tvalid <= n_m_axis_tvalid;
        end
    end

    //Status
    always_ff @ (posedge m_axi_aclk or negedge m_axi_aresetn) begin
        if (!m_axi_aresetn) begin
            sptr <= '0;
        end else if (we && !re && !full) begin   //Only write
            sptr <= sptr + 1;
        end else if (!we && re && !empty) begin  //Only read
            sptr <= sptr - 1;
        end
    end

endmodule

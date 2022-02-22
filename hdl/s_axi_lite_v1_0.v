
`timescale 1 ns / 1 ps

	module s_axi_lite_v1_0 #
	(
		// Users to add parameters here

		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface S_AXI
		parameter integer C_S_AXI_DATA_WIDTH	= 32,
		parameter integer C_S_AXI_ADDR_WIDTH	= 7
	)
	(
		// Users to add ports here
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg0,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg1,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg2,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg3,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg4,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg5,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg6,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg7,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg8,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg9,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg10,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg11,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg12,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg13,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg14,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg15,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg16,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg17,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg18,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg19,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg20,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg21,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg22,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg23,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg24,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg25,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg26,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg27,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg28,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg29,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg30,
      output wire [C_S_AXI_DATA_WIDTH-1:0] reg31,
		// User ports ends
		// Do not modify the ports beyond this line
      

		// Ports of Axi Slave Bus Interface S_AXI
		input wire  s_axi_aclk,
		input wire  s_axi_aresetn,
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] s_axi_awaddr,
		input wire [2 : 0] s_axi_awprot,
		input wire  s_axi_awvalid,
		output wire  s_axi_awready,
		input wire [C_S_AXI_DATA_WIDTH-1 : 0] s_axi_wdata,
		input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] s_axi_wstrb,
		input wire  s_axi_wvalid,
		output wire  s_axi_wready,
		output wire [1 : 0] s_axi_bresp,
		output wire  s_axi_bvalid,
		input wire  s_axi_bready,
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] s_axi_araddr,
		input wire [2 : 0] s_axi_arprot,
		input wire  s_axi_arvalid,
		output wire  s_axi_arready,
		output wire [C_S_AXI_DATA_WIDTH-1 : 0] s_axi_rdata,
		output wire [1 : 0] s_axi_rresp,
		output wire  s_axi_rvalid,
		input wire  s_axi_rready
	);
// Instantiation of Axi Bus Interface S_AXI
	s_axi_lite_v1_0_S_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
	) s_axi_lite_v1_0_S_AXI_inst (
      .reg0  (reg0 ),
      .reg1  (reg1 ),
      .reg2  (reg2 ),
      .reg3  (reg3 ),
      .reg4  (reg4 ),
      .reg5  (reg5 ),
      .reg6  (reg6 ),
      .reg7  (reg7 ),
      .reg8  (reg8 ),
      .reg9  (reg9 ),
      .reg10 (reg10),
      .reg11 (reg11),
      .reg12 (reg12),
      .reg13 (reg13),
      .reg14 (reg14),
      .reg15 (reg15),
      .reg16 (reg16),
      .reg17 (reg17),
      .reg18 (reg18),
      .reg19 (reg19),
      .reg20 (reg20),
      .reg21 (reg21),
      .reg22 (reg22),
      .reg23 (reg23),
      .reg24 (reg24),
      .reg25 (reg25),
      .reg26 (reg26),
      .reg27 (reg27),
      .reg28 (reg28),
      .reg29 (reg29),
      .reg30 (reg30),
      .reg31 (reg31),
		.S_AXI_ACLK(s_axi_aclk),
		.S_AXI_ARESETN(s_axi_aresetn),
		.S_AXI_AWADDR(s_axi_awaddr),
		.S_AXI_AWPROT(s_axi_awprot),
		.S_AXI_AWVALID(s_axi_awvalid),
		.S_AXI_AWREADY(s_axi_awready),
		.S_AXI_WDATA(s_axi_wdata),
		.S_AXI_WSTRB(s_axi_wstrb),
		.S_AXI_WVALID(s_axi_wvalid),
		.S_AXI_WREADY(s_axi_wready),
		.S_AXI_BRESP(s_axi_bresp),
		.S_AXI_BVALID(s_axi_bvalid),
		.S_AXI_BREADY(s_axi_bready),
		.S_AXI_ARADDR(s_axi_araddr),
		.S_AXI_ARPROT(s_axi_arprot),
		.S_AXI_ARVALID(s_axi_arvalid),
		.S_AXI_ARREADY(s_axi_arready),
		.S_AXI_RDATA(s_axi_rdata),
		.S_AXI_RRESP(s_axi_rresp),
		.S_AXI_RVALID(s_axi_rvalid),
		.S_AXI_RREADY(s_axi_rready)
	);

	// Add user logic here

	// User logic ends

	endmodule

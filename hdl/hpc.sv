`timescale 1ns/1ps
`include "hdr_macros.v"

module hpc #(
   NUM_PE              = 16,
   CFG_BW              = 32,
   CFG_NUM             = 32,
   ACT_INT_BW          = 8,
   ACT_FRA_BW          = 8,
   W_INT_BW            = 8,
   W_FRA_BW            = 8,
   NUM_LAYER_BW        = 2,
   LAYER_SIZE_BW       = 10,
   MEM_ACC_DEPTH_BW    = 9,  
   MEM_ACC_DEPTH_SL_BW = 8
   )(
   input  logic                                                        clk,
   input  logic                                                        rst_n,
   input  logic        [CFG_BW-1:0]                                    cfg            [CFG_NUM-1:0],
   // NZVL Input S_AXIS Interface
   input  logic                                                        s_nzv_axis_tvalid,
   output logic                                                        s_nzv_axis_tready,
   input  logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                   s_nzv_axis_tdata,
   input  logic                                                        s_nzv_axis_tlast,
   // Weight Input S_AXIS Interface  
   input  logic                                                        s_w_axis_tvalid,
   output logic                                                        s_w_axis_tready,
   input  logic signed [NUM_PE-1:0][W_INT_BW+W_FRA_BW-1:0]             s_w_axis_tdata,
   input  logic                                                        s_w_axis_tlast,
   // Output M_AXIS Interface
   output logic                                                        m_act_axis_tvalid,
   input  logic                                                        m_act_axis_tready,
   output logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]       m_act_axis_tdata,
   output logic                                                        m_act_axis_tlast,
   // H(t-1) Access Interface  
   input  logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]       din_hpt,
   output logic                                                        rd_en_hpt
   );
   
   //-----------------------------------------------------------
   // Local Parameters
   //-----------------------------------------------------------
   localparam ACT_BW = ACT_INT_BW + ACT_FRA_BW;
   localparam W_BW   = W_INT_BW + W_FRA_BW;
   localparam ACC_BW = ACT_BW + W_BW;
   
   //-----------------------
   //-- Logic
   //-----------------------
   
   // State Machine
   typedef enum logic [2:0]   {HPC_INIT = 3'h0,
                               HPC_IDLE = 3'h1,
                               HPC_MAC  = 3'h2,
                               HPC_ACT0 = 3'h3,
                               HPC_ACT1 = 3'h4,
                               HPC_ACT2 = 3'h5,
                               HPC_ACT3 = 3'h6,
                               HPC_ACT4 = 3'h7} HPC_STATE;
   
   // HPE Logic
   logic                                   hpe_en;
   logic                                   hpe_en_sig;
   logic                                   hpe_en_tanh;
   logic          [1:0]                    hpe_s0;
   logic          [1:0]                    hpe_s1;
   logic          [1:0]                    hpe_s2;
   logic                                   hpe_s3;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0] hpe_din_hpt;
   logic signed   [NUM_PE-1:0][W_BW-1:0]   hpe_din_w;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0] hpe_din_nzv;
   logic signed   [NUM_PE-1:0][ACC_BW-1:0] hpe_din_acc;
   logic signed   [NUM_PE-1:0][ACC_BW-1:0] hpe_dout_m;
   logic signed   [NUM_PE-1:0][ACT_BW-1:0] hpe_dout_ht;
   
   // MEM_ACC Logic
   logic                                    mem_acc_wr_en;
   logic                                    mem_acc_rd_en;
   logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0] mem_acc_addr_wr;
   logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0] mem_acc_addr_rd;
   logic signed   [NUM_PE-1:0][ACC_BW-1:0]  mem_acc_din;
   logic signed   [NUM_PE-1:0][ACC_BW-1:0]  mem_acc_dout;

   // Controller Logic
   logic hpe_mode;
   (* mark_debug = "true" *) HPC_STATE hpc_cs, hpc_ns;
   logic rst_n_ctrl;
   
   // Operand Valid Pipeline
   logic op_valid, r0_op_valid, r1_op_valid, r2_op_valid;
   logic n_s_nzv_axis_tvalid, r0_s_nzv_axis_tvalid, r1_s_nzv_axis_tvalid, r2_s_nzv_axis_tvalid;
   
   // Operand Last Pipeline
   logic w_last, r0_w_last, r1_w_last, r2_w_last;
   logic w_valid, r0_w_valid, r1_w_valid, r2_w_valid;
   logic nzv_last, r0_nzv_last, r1_nzv_last, r2_nzv_last;
   
   // X/H Phase Select
   logic sel_xh_rd;
   logic sel_xh_wr;
   logic en_sel_xh_rd;
   logic en_sel_xh_wr;
   
   //X/H Phase Detect Counter
   logic [1:0] cnt_phase;
   
   // ACT Address Counter
   logic en_cnt_addr_act;
   logic unsigned [LAYER_SIZE_BW:0] cnt_addr_act;
   
   // MEM ACC Read Address
   logic en_cnt_rd_addr_mem_acc;
   logic unsigned [MEM_ACC_DEPTH_BW-1:0]  cnt_rd_addr_mem_acc;
   
   // MEM ACC Write Address
   logic en_cnt_wr_addr_mem_acc;
   logic unsigned [MEM_ACC_DEPTH_BW-1:0]  cnt_wr_addr_mem_acc;
   
   // Activation Address Offset
   logic unsigned [MEM_ACC_DEPTH_SL_BW-1:0] addr_act_offset;
   
   // Layer Address Counter
   logic rst_n_cnt_layer;
   logic en_cnt_addr_layer;
   logic unsigned [NUM_LAYER_BW-1:0] cnt_addr_layer;
   
   // Configurations
   logic unsigned [NUM_LAYER_BW:0]  num_layer;
   
   //-----------------------
   //-- Configurations
   //-----------------------
   logic [LAYER_SIZE_BW:0]      curr_hid_size;
   logic [MEM_ACC_DEPTH_BW-1:0] gate_addr_range;
   always_ff @ (posedge clk) begin
      curr_hid_size <= cfg[CFG_IDX_NUM_LAYER+2+cnt_addr_layer][LAYER_SIZE_BW:0];
   end
   assign num_layer = cfg[CFG_IDX_NUM_LAYER][NUM_LAYER_BW:0];
   assign gate_addr_range = curr_hid_size/NUM_PE;
   
   //-----------------------
   //-- Connection
   //-----------------------
   assign hpe_din_w     = s_w_axis_tdata;
   assign hpe_din_acc   = mem_acc_dout;
   assign hpe_din_hpt   = din_hpt;
   assign mem_acc_din   = hpe_dout_m;
   assign m_act_axis_tdata = hpe_dout_ht;
   always_comb begin
      for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
         hpe_din_nzv[i] = s_nzv_axis_tdata;
      end
   end
   
   //-------------------------
   //-- Layer Address Counter
   //-------------------------
   always_ff @ (posedge clk) begin
      if (!rst_n | !rst_n_cnt_layer) begin
         cnt_addr_layer <= '0;
      end else if (en_cnt_addr_layer) begin
         cnt_addr_layer <= cnt_addr_layer + 1;
      end
   end
   
   //-----------------------
   //-- Pipeline Indicators
   //-----------------------
   
   // X/H Phase Select
   always_ff @ (posedge clk) begin
      if (!rst_n) begin
         sel_xh_rd <= '0;
         sel_xh_wr <= '0;
      end else begin
         if (en_sel_xh_rd)
            sel_xh_rd <= sel_xh_rd + 1;
         if (en_sel_xh_wr)
            sel_xh_wr <= sel_xh_wr + 1;
      end
   end

   // Operand Valid Pipeline
   always_comb begin
      n_s_nzv_axis_tvalid = s_nzv_axis_tvalid;
      if (s_nzv_axis_tdata == 0)
         n_s_nzv_axis_tvalid = '0;
      if (!s_nzv_axis_tvalid | !s_w_axis_tvalid) begin
         op_valid = '0;
      end else begin
         if (s_nzv_axis_tdata != 0) begin
            op_valid = '1;
         end else begin
            op_valid = '0;
         end
      end
   end
   //assign op_valid = s_nzv_axis_tvalid && s_w_axis_tvalid && (s_nzv_axis_tdata != 0);
   always_ff @ (posedge clk) begin
      if (!rst_n) begin
         r0_s_nzv_axis_tvalid <= '0;
         r1_s_nzv_axis_tvalid <= '0;
         r2_s_nzv_axis_tvalid <= '0;
         r0_op_valid <= '0;
         r1_op_valid <= '0;
         r2_op_valid <= '0;
      end else if (hpe_en) begin
         r0_op_valid <= op_valid;
         r1_op_valid <= r0_op_valid;
         r2_op_valid <= r1_op_valid;
         r0_s_nzv_axis_tvalid <= n_s_nzv_axis_tvalid;
         r1_s_nzv_axis_tvalid <= r0_s_nzv_axis_tvalid;
         r2_s_nzv_axis_tvalid <= r1_s_nzv_axis_tvalid;
      end
   end
   
   // NZV Last Pipeline
   assign nzv_last = s_nzv_axis_tlast;
   always_ff @ (posedge clk) begin
      if (!rst_n) begin
         r0_nzv_last <= '0;
         r1_nzv_last <= '0;
         r2_nzv_last <= '0;
      end else if (hpe_en) begin
         r0_nzv_last <= nzv_last;
         r1_nzv_last <= r0_nzv_last;
         r2_nzv_last <= r1_nzv_last;
      end
   end

   // Weight Flag Pipeline
   assign w_last = s_w_axis_tlast;
   assign w_valid = s_w_axis_tvalid;
   always_ff @ (posedge clk) begin
      if (!rst_n) begin
         r0_w_last <= '0;
         r1_w_last <= '0;
         r2_w_last <= '0;
         r0_w_valid <= '0;
         r1_w_valid <= '0;
         r2_w_valid <= '0;
      end else if (hpe_en) begin
         r0_w_last <= w_last;
         r1_w_last <= r0_w_last;
         r2_w_last <= r1_w_last;
         r0_w_valid <= w_valid;
         r1_w_valid <= r0_w_valid;
         r2_w_valid <= r1_w_valid;
      end
   end
   
   // ACT Address Counter
   always_ff @ (posedge clk) begin
      if (!rst_n | !rst_n_ctrl) begin
         cnt_addr_act <= '0;
      end else if (en_cnt_addr_act) begin // MAC Mode
         cnt_addr_act <= cnt_addr_act + 1;
      end
   end
   
   // MEM ACC Read Address Counter
   always_ff @ (posedge clk) begin
      if (!rst_n | !rst_n_ctrl) begin
         cnt_rd_addr_mem_acc <= '0;
      end else if (en_cnt_rd_addr_mem_acc) begin // MAC Mode
         if (w_last & w_valid) begin
            cnt_rd_addr_mem_acc <= '0;
         end else begin
            cnt_rd_addr_mem_acc <= cnt_rd_addr_mem_acc + 1;
         end
      end
   end
   
   // MEM ACC Write Address Counter
   always_ff @ (posedge clk) begin
      if (!rst_n | !rst_n_ctrl) begin
         cnt_wr_addr_mem_acc <= '0;
      end else if (en_cnt_wr_addr_mem_acc) begin // MAC Mode
         if (r2_w_last & r2_w_valid) begin
            cnt_wr_addr_mem_acc <= '0;
         end else begin
            cnt_wr_addr_mem_acc <= cnt_wr_addr_mem_acc + 1;
         end
      end 
   end
   
   // X/H Phase Detect Counter
   always_ff @ (posedge clk) begin
      if (!rst_n | cnt_phase == 2'h2) begin
         cnt_phase <= '0;
      end else if ((r2_w_last & r2_w_valid & r2_nzv_last) || (r2_nzv_last & !r2_s_nzv_axis_tvalid)) begin
         cnt_phase <= cnt_phase + 1;
      end
   end
   
   //--------------------------
   //-- MEM ACC Address Encode
   //--------------------------
   
   // Read Address
   always_comb begin
      mem_acc_addr_rd = '0;
      if (!hpe_mode) begin
         if (!sel_xh_rd) begin
            if (cnt_rd_addr_mem_acc < 2*gate_addr_range) begin
               mem_acc_addr_rd = cnt_rd_addr_mem_acc;
            end else begin
               mem_acc_addr_rd = cnt_rd_addr_mem_acc + curr_hid_size/NUM_PE;
            end
         end else begin
            mem_acc_addr_rd = cnt_rd_addr_mem_acc;
         end
      end else begin
         mem_acc_addr_rd = addr_act_offset + cnt_addr_act;
      end
   end
   
   // Write Address
   always_comb begin
      mem_acc_addr_wr = '0;
      if (!sel_xh_wr) begin
         if (cnt_wr_addr_mem_acc < 2*curr_hid_size/NUM_PE) begin
            mem_acc_addr_wr = cnt_wr_addr_mem_acc;
         end else begin
            mem_acc_addr_wr = cnt_wr_addr_mem_acc + curr_hid_size/NUM_PE;
         end
      end else begin
         mem_acc_addr_wr = cnt_wr_addr_mem_acc;
      end
   end
   
   //-----------------------
   //-- Controller
   //-----------------------
   
   // Sequencial Logic
   always_ff @ (posedge clk) begin
      if (!rst_n) begin
         hpc_cs <= HPC_INIT;
      end else begin
         hpc_cs <= hpc_ns;
      end
   end
   
   // Combinational Logic
   always_comb begin
      hpc_ns                 = HPC_INIT;
      rst_n_ctrl             = '1;
      rst_n_cnt_layer        = '1;
      s_nzv_axis_tready         = '0;
      s_w_axis_tready         = '0;
      mem_acc_rd_en          = '0;
      mem_acc_wr_en          = '0;
      rd_en_hpt              = '0;
      hpe_mode               = '0;
      hpe_en                 = '0;
      hpe_en_sig             = '0;
      hpe_en_tanh            = '0;
      hpe_s0                 = '0;
      hpe_s1                 = '0;
      hpe_s2                 = '0;
      hpe_s3                 = '0;
      en_cnt_addr_act        = '0;
      en_cnt_rd_addr_mem_acc = '0;
      en_cnt_wr_addr_mem_acc = '0;
      en_cnt_addr_layer      = '0;
      addr_act_offset        = '0;
      en_sel_xh_rd           = '0;
      en_sel_xh_wr           = '0;
      m_act_axis_tvalid         = '0;
      m_act_axis_tlast          = '0;
      case(hpc_cs)
         // Initialization
         HPC_INIT: //------------------------------------------------
         begin
            hpc_ns            = HPC_MAC;
            rst_n_ctrl        = '0;
         end
         
         // IDLE
         HPC_IDLE: //------------------------------------------------
         begin
            hpc_ns            = HPC_IDLE;
            if (op_valid) begin
               hpc_ns            = HPC_MAC;
               hpe_en            = '1;
               hpe_s0            = 2'b10;
               hpe_s1            = 2'b00;
               hpe_s2            = 2'b01;
               mem_acc_rd_en     = op_valid;
               en_cnt_rd_addr_mem_acc = op_valid;
               s_w_axis_tready    = '1;
            end
         end
         
         // Reset
         HPC_MAC: //------------------------------------------------
         begin
            hpc_ns                 = HPC_MAC;
            hpe_en                 = '1;
            hpe_s0                 = 2'b10;
            hpe_s1                 = 2'b00;
            hpe_s2                 = 2'b01;
            s_nzv_axis_tready         = !s_nzv_axis_tvalid;
            s_w_axis_tready         = n_s_nzv_axis_tvalid;
            mem_acc_rd_en          = op_valid;
            mem_acc_wr_en          = r2_op_valid;
            en_sel_xh_rd           = (nzv_last & w_last & w_valid) | (nzv_last & !n_s_nzv_axis_tvalid);
            en_sel_xh_wr           = (r2_nzv_last & r2_w_last & r2_w_valid) | (r2_nzv_last & !r2_s_nzv_axis_tvalid);
            en_cnt_rd_addr_mem_acc = op_valid;
            en_cnt_wr_addr_mem_acc = r2_op_valid;
            
            if ((w_last & w_valid) | (nzv_last & !n_s_nzv_axis_tvalid)) begin
               s_nzv_axis_tready = '1;
            end
            
            if (cnt_phase == 1 && ((r2_nzv_last & r2_w_last & r2_w_valid) | (r2_nzv_last & !r2_s_nzv_axis_tvalid))) begin
               hpc_ns  = HPC_ACT0;
            end     
         end
         
         // Activation Stage 0 or 5
         HPC_ACT0: //------------------------------------------------
         begin
            hpc_ns          = HPC_ACT1;
            hpe_mode        = '1;
            hpe_en          = '1;
            hpe_en_tanh     = '1;
            hpe_s0          = 2'b11;
            hpe_s1          = 2'b01;
            mem_acc_rd_en   = '1;
            addr_act_offset = '0;
            
            if (cnt_addr_act >= gate_addr_range) begin
               mem_acc_rd_en = '0;
            end
            
            if (cnt_addr_act > 1) begin
               m_act_axis_tvalid = '1;
               if (m_act_axis_tready) begin
                  hpe_en        = '1;
                  hpe_en_tanh   = '1;
                  mem_acc_rd_en = '1;
                  hpc_ns = HPC_ACT1;
               end else begin
                  hpe_en        = '0;
                  hpe_en_tanh   = '0;
                  mem_acc_rd_en = '0;
                  hpc_ns = HPC_ACT0;
               end
            end
            
            if (cnt_addr_act == gate_addr_range+1) begin
               m_act_axis_tlast      = '1;
               en_cnt_addr_layer = '1;
               hpc_ns = HPC_INIT;
               if (cnt_addr_layer == num_layer-1) begin
                  rst_n_cnt_layer = '0;
               end
            end
         end
         
         // Activation Stage 1 or 6
         HPC_ACT1: //------------------------------------------------
         begin
            hpc_ns          = HPC_ACT2;
            hpe_mode        = '1;
            hpe_en          = '1;
            hpe_en_sig      = '1;
            hpe_s0          = 2'b00;
            hpe_s1          = 2'b10;
            hpe_s2          = 2'b00;
            mem_acc_rd_en   = '1;
            addr_act_offset = 2*gate_addr_range;
            
            if (cnt_addr_act >= gate_addr_range) begin
               mem_acc_rd_en = '0;
            end
         end
         
         // Activation Stage 2 or 7
         HPC_ACT2: //------------------------------------------------
         begin
            hpc_ns          = HPC_ACT3;
            hpe_mode        = '1;
            hpe_en          = '1;
            hpe_en_sig      = '0;
            hpe_s2          = 2'b00;
            mem_acc_rd_en   = '1;
            addr_act_offset = gate_addr_range;
            
            if (cnt_addr_act >= gate_addr_range) begin
               mem_acc_rd_en = '0;
            end
         end
         
         // Activation Stage 3 or 8
         HPC_ACT3: //------------------------------------------------
         begin
            hpc_ns          = HPC_ACT4;
            hpe_mode        = '1;
            hpe_en          = '1;
            hpe_en_sig      = '1;
            hpe_s0          = 2'b01;
            hpe_s1          = 2'b11;
            mem_acc_rd_en   = '1;
            addr_act_offset = 3*gate_addr_range;
            en_cnt_addr_act = '1;
            
            if (cnt_addr_act >= gate_addr_range) begin
               mem_acc_rd_en = '0;
            end
         end
         
         // Activation Stage 4 or 9
         HPC_ACT4: //------------------------------------------------
         begin
            hpc_ns          = HPC_ACT0;
            hpe_mode        = '1;
            hpe_en          = '1;
            hpe_s2          = 2'b10;
            hpe_s3          = 1'b1;
            rd_en_hpt       = '1;
            addr_act_offset = 3*gate_addr_range;
            
            if (cnt_addr_act >= gate_addr_range+1) begin
               rd_en_hpt = '0;
            end
            
         end
         
         default: //-------------------------------------------
         begin
            hpc_ns          = HPC_INIT;
            rst_n_ctrl             = '1;
            s_nzv_axis_tready         = '0;
            s_w_axis_tready         = '0;
            mem_acc_rd_en          = '0;
            mem_acc_wr_en          = '0;
            rd_en_hpt              = '0;
            hpe_mode               = '0;
            hpe_en                 = '0;
            hpe_en_sig             = '0;
            hpe_en_tanh            = '0;
            hpe_s0                 = '0;
            hpe_s1                 = '0;
            hpe_s2                 = '0;
            hpe_s3                 = '0;
            en_cnt_addr_act        = '0;
            en_cnt_rd_addr_mem_acc = '0;
            en_cnt_wr_addr_mem_acc = '0;
            addr_act_offset        = '0;
            en_sel_xh_rd           = '0;
            en_sel_xh_wr           = '0;
            m_act_axis_tvalid          = '0;
            m_act_axis_tlast           = '0;
         end
      endcase
   end
   
   //-----------------------
   //-- Instantiation
   //-----------------------
   
   // Instantiate Heterogeneous Process Element
   hpe # (
      .NUM_PE            (NUM_PE    ),
      .ACT_INT_BW        (ACT_INT_BW),
      .ACT_FRA_BW        (ACT_FRA_BW),
      .W_INT_BW          (W_INT_BW  ),
      .W_FRA_BW          (W_FRA_BW  )
   ) i_hpe (
      `ifdef DEBUG //----------------------------------------------
      .sigmoid_deb_out (sigmoid_deb_out),
      .tanh_deb_out (tanh_deb_out),
      .mul_deb_out (mul_deb_out),
      .add_deb_out (add_deb_out),
      .add_act_deb_out (add_act_deb_out),
      `endif //--------------------------------------------------------
      .clk     (clk        ),
      .rst_n   (rst_n      ),
      .en      (hpe_en     ),
      .en_sig  (hpe_en_sig ),
      .en_tanh (hpe_en_tanh),
      .s0      (hpe_s0     ),
      .s1      (hpe_s1     ),
      .s2      (hpe_s2     ),
      .s3      (hpe_s3     ),
      .din_hpt (hpe_din_hpt),
      .din_w   (hpe_din_w  ),
      .din_nzv (hpe_din_nzv),
      .din_acc (hpe_din_acc),
      .dout_m  (hpe_dout_m ),
      .dout_ht (hpe_dout_ht)
   );
   
   // Instantiate Accumulation Memory Cluster
   mem_acc # (
      .NUM_PE              (NUM_PE             ),
      .ACC_BW              (ACC_BW             ),
      .NUM_LAYER_BW        (NUM_LAYER_BW       ),    
      .MEM_ACC_DEPTH_BW    (MEM_ACC_DEPTH_BW   ),  
      .MEM_ACC_DEPTH_SL_BW (MEM_ACC_DEPTH_SL_BW)
   ) i_mem_acc (
      .clk        (clk               ),
   `ifdef SIM_DEBUG //-----------------------
      .rst_n      (rst_n             ),
   `endif //---------------------------------
      .wr_en      (mem_acc_wr_en     ),
      .rd_en      (mem_acc_rd_en     ),
      .addr_layer (cnt_addr_layer    ),
      .addr_wr    (mem_acc_addr_wr   ),
      .addr_rd    (mem_acc_addr_rd   ),
      .din        (mem_acc_din       ),
      .dout       (mem_acc_dout      )
   );
   
endmodule


`include "hdr_macros.v"

module hpe # (
    NUM_PE     = 4,
    ACT_INT_BW = 8,
    ACT_FRA_BW = 8,
    W_INT_BW   = 8,
    W_FRA_BW   = 8
    )(
    `ifdef DEBUG //----------------------------------------------
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                   sigmoid_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                   tanh_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] mul_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] add_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                   add_act_deb_out,
    `endif //--------------------------------------------------------
    input  logic                                                                    clk,
    input  logic                                                                    rst_n,
    input  logic                                                                    en,
    input  logic                                                                    en_sig,
    input  logic                                                                    en_tanh,
    input  logic        [1:0]                                                       s0,
    input  logic        [1:0]                                                       s1,
    input  logic        [1:0]                                                       s2,
    input  logic                                                                    s3,
    input  logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   din_hpt,
    input  logic signed [NUM_PE-1:0][(W_INT_BW+W_FRA_BW)-1:0]                       din_w,
    input  logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   din_nzv,
    input  logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] din_acc,
    output logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] dout_m,
    output logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   dout_ht 
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

    // Accumulation Adder
    logic signed [NUM_PE-1:0][ACC_BW-1:0] add_op0;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] add_op1;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r2_add;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r3_add_out;

    // Sigmoid
    logic signed [NUM_PE-1:0][ACT_BW-1:0] sig_op;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] sig_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_sig_op;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_sig_out;

    // Tanh
    logic signed [NUM_PE-1:0][ACT_BW-1:0] tanh_op    ;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] tanh_out   ;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_tanh_op ;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r0_tanh_out;

    // Accumulation Input Rounding
    logic signed [NUM_PE-1:0][ACC_BW-1:0] din_acc_rnd        ;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] din_acc_rnd_q_act  ;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_din_acc_rnd_q_act;

    // MAC Adder Output Rounding
    logic signed [NUM_PE-1:0][ACT_BW-1:0] add_out_rnd_q_act;

    // Multiplexers
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1_in_nzv;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1_in_sig;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1_in_acc;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1_in_act;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] mux_s2_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_1;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_sig_n;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_act_0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_act_1;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_1;

    // MAC Multiplier
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r0_op0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r0_op1;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r1_mul;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r1_mul_out;

    // Activation Adder
    logic signed [NUM_PE-1:0][ACT_BW-1:0] add_act_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_add_act_out;

    //-----------------------
    //-- Debug
    //-----------------------
    `ifdef DEBUG //----------------------------------------------
    assign sigmoid_deb_out = sig_out[1];
    assign tanh_deb_out = tanh_out[1];
    assign mul_deb_out = r1_mul[1];
    assign add_deb_out = r2_add[1];
    assign add_act_deb_out = add_act_out[1];
    `endif //--------------------------------------------------------

    //-----------------------
    //-- I/O
    //-----------------------
    assign dout_m = r3_add_out;
    assign dout_ht = r_add_act_out;

    //-----------------------
    //-- Rounding
    //-----------------------
    // din_acc_rnd_q_act[i] = (din_acc[i]
    //                            + { 
    //                                  {(ACC_BW-W_FRA_BW){1'b0}},
    //                                  din_acc[i][W_FRA_BW] ^ din_acc[i][ACC_BW-1],
    //                                  {(W_FRA_BW-1){din_acc[i][W_FRA_BW] ~^ din_acc[i][ACC_BW-1]}}
    //                               }) >>> W_FRA_BW;
    always_comb begin
        for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
            // Round Accumulation Input To the Precision of Activations
            din_acc_rnd_q_act[i] = (din_acc[i]
                                + { 
                                    {(ACC_BW-W_FRA_BW){1'b0}},
                                    din_acc[i][W_FRA_BW],
                                    {(W_FRA_BW-1){!din_acc[i][W_FRA_BW]}}
                                    }) >>> W_FRA_BW;
            
            
            // Round Adder Output to Activation Precision
            //add_out_rnd_q_act[i] = bitsra_conv(r2_add[i], ACT_FRA_BW);
            add_out_rnd_q_act[i] = (r2_add[i]
                                + { 
                                    {(ACC_BW-ACT_FRA_BW){1'b0}},
                                    r2_add[i][ACT_FRA_BW],
                                    {(ACT_FRA_BW-1){!r2_add[i][ACT_FRA_BW]}}
                                    }) >>> ACT_FRA_BW;
        end
    end

    //-----------------------
    //-- Multiplexers
    //-----------------------
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            r_din_acc_rnd_q_act <= '0;
            mux_s3_in_act_0     <= '0;
            mux_s3_in_act_1     <= '0;
        end else if (en) begin
            r_din_acc_rnd_q_act <= din_acc_rnd_q_act;
            mux_s3_in_act_0 <= add_out_rnd_q_act;
            mux_s3_in_act_1 <= mux_s3_in_act_0;
        end
    end

    always_comb begin
        // Multiplexer S0
        case(s0)
            2'b00 : mux_s0 = tanh_out;   // c
            2'b01 : mux_s0 = r_sig_out; // r
            2'b10 : begin
                for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
                    mux_s0[i] = $signed(din_w[i]);
                end
            end
            2'b11 : mux_s0 = din_hpt;    // h-1
        endcase
        // Multiplexer S1
        case(s1)
            2'b00 : mux_s1 = din_nzv;
            2'b01 : mux_s1 = r_sig_out;
            2'b10 : mux_s1 = r_add_act_out;
            2'b11 : mux_s1 = r_din_acc_rnd_q_act;
        endcase
        // Multiplexer S2
        case(s2)
            // Allow outputs of multiplication results
            2'b00 : mux_s2_out = '0;
            // Use original accumulation precision for MxV
            2'b01 : mux_s2_out = din_acc;
            // Align decimal points between Wcx and r*Wch
            2'b10 : begin
                for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
                    mux_s2_out[i] = $signed(din_acc[i]) * (2**(ACT_FRA_BW-W_FRA_BW));
                end
            end
            default: mux_s2_out = '0;
        endcase
        // Multiplexer S3
        for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
            mux_s3_in_1    [i] = 2**ACT_FRA_BW;
            mux_s3_in_sig_n[i] = -r_sig_out[i]; // Get negative sigmoid
        end
        if (!s3) begin
            mux_s3_0 = mux_s3_in_1;
            mux_s3_1 = mux_s3_in_sig_n;
        end else begin
            mux_s3_0 = mux_s3_in_act_0;
            mux_s3_1 = mux_s3_in_act_1;
        end
    end

    //----------------------------------------
    //-- MAC Unit
    //----------------------------------------
    assign r0_op0 = mux_s0;
    assign r0_op1 = mux_s1;

    // Multiply-Add Unit
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            r1_mul     <= '0;
            r2_add     <= '0;
            r3_add_out <= '0;
        end else if (en) begin
            for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
                r1_mul[i]     <= $signed(r0_op0[i]) * $signed(r0_op1[i]);
                r2_add[i]     <= $signed(r1_mul[i]) + $signed(mux_s2_out[i]);
                r3_add_out[i] <= r2_add[i];
            end
        end
    end

    // Activation Adder
    always_comb begin
        for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
            add_act_out[i] = $signed(mux_s3_0[i]) + $signed(mux_s3_1[i]);
        end
    end

    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
                r_add_act_out[i] <= '0;
            end
        end else if (en) begin
            r_add_act_out <= add_act_out;
        end
    end

    //----------------------------------------
    //-- Non-Linear Functions
    //----------------------------------------
    assign sig_op = din_acc_rnd_q_act;
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            r_sig_op <= '0;
            r_sig_out <= '0;
        end else if (en) begin
            if (en_sig) r_sig_op <= sig_op;
            r_sig_out <= sig_out;
        end
    end

    sigmoid_lut #(
        .NUM_PE     (NUM_PE),
        .ACT_INT_BW (ACT_INT_BW),
        .ACT_FRA_BW (ACT_FRA_BW)
    ) i_sigmoid_lut (
        r_sig_op,
        sig_out
    );

    assign tanh_op = add_out_rnd_q_act;
    always_ff @ (posedge clk) begin
        if (!rst_n) begin
            for (int unsigned i = 0; i < NUM_PE; i = i + 1) begin
                r_tanh_op[i] <= '0;
            end
        end else if (en_tanh & en) begin
            r_tanh_op <= tanh_op;
        end
    end

    tanh_lut #(
        .NUM_PE     (NUM_PE),
        .ACT_INT_BW (ACT_INT_BW),
        .ACT_FRA_BW (ACT_FRA_BW)
    ) i_tanh_lut (
        r_tanh_op,
        tanh_out
    );

endmodule

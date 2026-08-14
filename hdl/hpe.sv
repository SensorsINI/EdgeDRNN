`default_nettype none

/**
 * Heterogeneous Processing Element (HPE) array.
 *
 * This module is the arithmetic datapath shown in paper Figs. 5 and 7.  The
 * same NUM_PE lanes are time-multiplexed between two jobs:
 *
 *   1. sparse matrix-vector update: M[row] += delta * W[row, column]
 *   2. GRU activation generation: calculate r, u, c, then h from the four
 *      accumulation-memory regions in Eq. (3)
 *
 * The HPC controller selects operands with s0..s3.  Arithmetic is fixed-point
 * and deliberately wraps to the destination width, matching the quantized
 * PyTorch exporter.  The pipeline advances only when en is high; deasserting
 * en freezes every stage so AXI backpressure cannot corrupt an operation.
 *
 * Fixed-point formats used by the default design:
 *   activation/state : signed Q7.8  (ACT_BW = 16)
 *   weight           : signed Q0.7  (W_BW   = 8)
 *   accumulator      : signed product width (ACC_BW = 24)
 */
module hpe #(
    parameter int unsigned NUM_PE     = 4,
    parameter int unsigned ACT_INT_BW = 8,
    parameter int unsigned ACT_FRA_BW = 8,
    parameter int unsigned W_INT_BW   = 8,
    parameter int unsigned W_FRA_BW   = 8
) (
`ifdef DEBUG
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                    sigmoid_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                    tanh_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] mul_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] add_deb_out,
    output logic signed [(ACT_INT_BW+ACT_FRA_BW)-1:0]                    add_act_deb_out,
`endif
    input  wire logic                                                                    clk,
    input  wire logic                                                                    rst_n,
    input  wire logic                                                                    en,
    input  wire logic                                                                    en_sig,
    input  wire logic                                                                    en_tanh,
    input  wire logic        [1:0]                                                       s0,
    input  wire logic        [1:0]                                                       s1,
    input  wire logic        [1:0]                                                       s2,
    input  wire logic                                                                    s3,
    input  wire logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   din_hpt,
    input  wire logic signed [NUM_PE-1:0][(W_INT_BW+W_FRA_BW)-1:0]                       din_w,
    input  wire logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   din_nzv,
    input  wire logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] din_acc,
    output logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW+W_INT_BW+W_FRA_BW)-1:0] dout_m,
    output logic signed [NUM_PE-1:0][(ACT_INT_BW+ACT_FRA_BW)-1:0]                   dout_ht
);
    timeunit 1ns;
    timeprecision 1ps;

    localparam int unsigned ACT_BW = ACT_INT_BW + ACT_FRA_BW;
    localparam int unsigned W_BW   = W_INT_BW + W_FRA_BW;
    localparam int unsigned ACC_BW = ACT_BW + W_BW;

    // Three-stage multiply/add pipeline.  r3_add_out is written back to the
    // accumulation memory by the HPC controller.
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r2_add;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r3_add_out;

    // Nonlinear activation signals.  LUT inputs are registered so every HPE
    // lane sees the same controller-selected activation phase.
    logic signed [NUM_PE-1:0][ACT_BW-1:0] sig_op;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] sig_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_sig_op;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_sig_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] tanh_op;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] tanh_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_tanh_op;

    // Rounding is performed at two different binary-point boundaries.
    logic signed [NUM_PE-1:0][ACT_BW-1:0] din_acc_rnd_q_act;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_din_acc_rnd_q_act;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] add_out_rnd_q_act;

    // Controller-selected operands.  See the selection table immediately
    // above the multiplexer logic below.
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s1;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] mux_s2_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_1;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_sig_n;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_act_0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_in_act_1;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] mux_s3_1;

    logic signed [NUM_PE-1:0][ACT_BW-1:0] r0_op0;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r0_op1;
    logic signed [NUM_PE-1:0][ACC_BW-1:0] r1_mul;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] add_act_out;
    logic signed [NUM_PE-1:0][ACT_BW-1:0] r_add_act_out;

`ifdef DEBUG
    assign sigmoid_deb_out = sig_out[1];
    assign tanh_deb_out = tanh_out[1];
    assign mul_deb_out = r1_mul[1];
    assign add_deb_out = r2_add[1];
    assign add_act_deb_out = add_act_out[1];
`endif

    assign dout_m  = r3_add_out;
    assign dout_ht = r_add_act_out;

    // Round-half-away-from-zero behavior inherited from the hardware export.
    // First align an accumulator to activation precision by removing the
    // weight fractional bits; later remove ACT_FRA_BW after an activation
    // multiply.  Assignment to ACT_BW is an intentional two's-complement
    // wrap and is part of the historical bit-exact numerical contract.
    /* verilator lint_off WIDTHTRUNC */
    always_comb begin
        for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
            din_acc_rnd_q_act[pe] = (din_acc[pe]
                + {{(ACC_BW-W_FRA_BW){1'b0}},
                   din_acc[pe][W_FRA_BW],
                   {(W_FRA_BW-1){!din_acc[pe][W_FRA_BW]}}}) >>> W_FRA_BW;

            add_out_rnd_q_act[pe] = (r2_add[pe]
                + {{(ACC_BW-ACT_FRA_BW){1'b0}},
                   r2_add[pe][ACT_FRA_BW],
                   {(ACT_FRA_BW-1){!r2_add[pe][ACT_FRA_BW]}}}) >>> ACT_FRA_BW;
        end
    end
    /* verilator lint_on WIDTHTRUNC */

    // Delay activation operands to line up with the multiplier/adder stages.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_din_acc_rnd_q_act <= '0;
            mux_s3_in_act_0     <= '0;
            mux_s3_in_act_1     <= '0;
        end else if (en) begin
            r_din_acc_rnd_q_act <= din_acc_rnd_q_act;
            mux_s3_in_act_0     <= add_out_rnd_q_act;
            mux_s3_in_act_1     <= mux_s3_in_act_0;
        end
    end

    /*
     * HPE operand selection (controlled by the HPC activation schedule):
     *
     * s0: 00=tanh(c), 01=sigmoid(r), 10=weight, 11=previous h
     * s1: 00=delta,   01=sigmoid result, 10=activation result,
     *     11=rounded accumulation memory
     * s2: 00=zero, 01=unscaled accumulation, 10=binary-point-aligned acc
     * s3: 0=(1-u)+u*c path, 1=two delayed activation terms
     */
    always_comb begin
        case (s0)
            2'b00: mux_s0 = tanh_out;
            2'b01: mux_s0 = r_sig_out;
            2'b10: begin
                for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
                    mux_s0[pe] = {
                        {(ACT_BW-W_BW){din_w[pe][W_BW-1]}},
                        din_w[pe]
                    };
                end
            end
            2'b11: mux_s0 = din_hpt;
            default: mux_s0 = '0;
        endcase

        case (s1)
            2'b00: mux_s1 = din_nzv;
            2'b01: mux_s1 = r_sig_out;
            2'b10: mux_s1 = r_add_act_out;
            2'b11: mux_s1 = r_din_acc_rnd_q_act;
            default: mux_s1 = '0;
        endcase

        case (s2)
            2'b00: mux_s2_out = '0;
            2'b01: mux_s2_out = din_acc;
            2'b10: begin
                for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
                    mux_s2_out[pe] = $signed(din_acc[pe]) * (2**(ACT_FRA_BW-W_FRA_BW));
                end
            end
            default: mux_s2_out = '0;
        endcase

        for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
            mux_s3_in_1[pe]     = 2**ACT_FRA_BW;
            mux_s3_in_sig_n[pe] = -r_sig_out[pe];
        end
        if (!s3) begin
            mux_s3_0 = mux_s3_in_1;
            mux_s3_1 = mux_s3_in_sig_n;
        end else begin
            mux_s3_0 = mux_s3_in_act_0;
            mux_s3_1 = mux_s3_in_act_1;
        end
    end

    assign r0_op0 = mux_s0;
    assign r0_op1 = mux_s1;

    // Each lane implements a three-register multiply-add pipe.  In sparse-MAC
    // mode the first operand is a weight and the second is the broadcast
    // non-zero delta.  Activation mode reuses the same multiplier for gates.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r1_mul     <= '0;
            r2_add     <= '0;
            r3_add_out <= '0;
        end else if (en) begin
            for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
                r1_mul[pe]     <= $signed(r0_op0[pe]) * $signed(r0_op1[pe]);
                r2_add[pe]     <= $signed(r1_mul[pe]) + $signed(mux_s2_out[pe]);
                r3_add_out[pe] <= r2_add[pe];
            end
        end
    end

    // A second, activation-width adder forms (1-u), the candidate blend, and
    // the final hidden-state expression while the main MAC remains reusable.
    always_comb begin
        for (int unsigned pe = 0; pe < NUM_PE; pe++) begin
            add_act_out[pe] = $signed(mux_s3_0[pe]) + $signed(mux_s3_1[pe]);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_add_act_out <= '0;
        end else if (en) begin
            r_add_act_out <= add_act_out;
        end
    end

    // Sigmoid path for reset/update gates.
    assign sig_op = din_acc_rnd_q_act;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_sig_op  <= '0;
            r_sig_out <= '0;
        end else if (en) begin
            if (en_sig) begin
                r_sig_op <= sig_op;
            end
            r_sig_out <= sig_out;
        end
    end

    sigmoid_lut #(
        .NUM_PE     (NUM_PE),
        .ACT_INT_BW (ACT_INT_BW),
        .ACT_FRA_BW (ACT_FRA_BW)
    ) i_sigmoid_lut (
        .in  (r_sig_op),
        .out (sig_out)
    );

    // Tanh path for the candidate state.
    assign tanh_op = add_out_rnd_q_act;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_tanh_op <= '0;
        end else if (en_tanh && en) begin
            r_tanh_op <= tanh_op;
        end
    end

    tanh_lut #(
        .NUM_PE     (NUM_PE),
        .ACT_INT_BW (ACT_INT_BW),
        .ACT_FRA_BW (ACT_FRA_BW)
    ) i_tanh_lut (
        .in  (r_tanh_op),
        .out (tanh_out)
    );

`ifndef SYNTHESIS
    initial begin
        assert (NUM_PE > 0) else $fatal(1, "NUM_PE must be positive");
        assert (ACT_BW >= W_BW)
            else $fatal(1, "Weight width must not exceed activation width");
        assert (W_FRA_BW > 0 && ACT_FRA_BW > 0)
            else $fatal(1, "Fractional widths must be positive");
        assert (ACT_FRA_BW >= W_FRA_BW)
            else $fatal(1, "HPE alignment assumes ACT_FRA_BW >= W_FRA_BW");
    end
`endif

endmodule

`default_nettype wire

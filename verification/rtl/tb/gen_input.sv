`timescale 1ns/1ps

module gen_input #(
    parameter integer NUM_PE = 8,
    parameter integer ACT_BW = 16,
    parameter integer NUM_STIM = 1,
    parameter STIM_FILE = "edgedrnn_tb_stim.txt"
) (
    input  logic                     s_axi_aclk,
    input  logic                     s_axi_aresetn,
    output logic                     m_axis_tvalid,
    input  logic                     m_axis_tready,
    output logic [NUM_PE*ACT_BW-1:0] m_axis_tdata,
    output logic                     m_axis_tlast
);

    typedef enum logic [1:0] {IG_IDLE, IG_READ, IG_END} input_state_t;
    input_state_t current_state, next_state;

    logic [NUM_PE*ACT_BW-1:0] memory [0:NUM_STIM-1];
    longint read_count;
    longint wait_count;
    logic read_enable;
    logic wait_count_enable;

    assign m_axis_tdata = (read_count < NUM_STIM) ? memory[read_count] : '0;

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            read_count <= 0;
            wait_count <= 0;
        end else begin
            if (read_enable) begin
                read_count <= read_count + 1;
            end
            if (wait_count_enable) begin
                wait_count <= wait_count + 1;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            current_state <= IG_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    always_comb begin
        next_state = IG_IDLE;
        read_enable = 1'b0;
        m_axis_tvalid = 1'b0;
        m_axis_tlast = 1'b0;
        wait_count_enable = 1'b0;
        case (current_state)
            IG_IDLE: begin
                next_state = IG_IDLE;
                wait_count_enable = 1'b1;
                if (wait_count > 1000) begin
                    m_axis_tvalid = s_axi_aresetn;
                    if (m_axis_tready) begin
                        next_state = IG_READ;
                        read_enable = 1'b1;
                    end
                end
            end
            IG_READ: begin
                next_state = IG_READ;
                m_axis_tvalid = 1'b1;
                read_enable = m_axis_tready;
                if (read_enable) begin
                    if (read_count == NUM_STIM - 1) begin
                        next_state = IG_END;
                    end else begin
                        next_state = IG_IDLE;
                    end
                end
            end
            IG_END: begin
                next_state = IG_END;
            end
            default: begin
                next_state = IG_IDLE;
            end
        endcase
    end

    initial begin
        if (NUM_STIM <= 0) begin
            $fatal(1, "NUM_STIM must be positive");
        end
        $readmemh(STIM_FILE, memory);
    end

endmodule

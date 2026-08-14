`timescale 1ns/1ps

module gen_weight #(
    parameter integer NUM_PE = 8,
    parameter integer W_BW = 8,
    parameter integer NUM_PARAMS = 1,
    parameter PARAM_FILE = "edgedrnn_tb_params.txt"
) (
    input  logic                   s_axi_aclk,
    input  logic                   s_axi_aresetn,
    input  logic                   s_axis_tvalid,
    output logic                   s_axis_tready,
    input  logic [79:0]            s_axis_tdata,
    input  logic                   s_axis_tlast,
    output logic                   m_axis_tvalid,
    input  logic                   m_axis_tready,
    output logic [NUM_PE*W_BW-1:0] m_axis_tdata,
    output logic                   m_axis_tlast
);

    typedef enum logic [1:0] {WMEM_IDLE, WMEM_READ} weight_state_t;
    weight_state_t current_state, next_state;

    logic [NUM_PE*W_BW-1:0] memory [0:NUM_PARAMS-1];
    longint burst_length;
    longint burst_length_bytes;
    logic load_pointer_n;
    longint read_beat_count;
    longint read_address_count;
    logic read_address_enable;
    longint read_address;
    logic unsigned [31:0] random_seed;

    always_comb begin
        m_axis_tdata = (read_address_count < NUM_PARAMS) ? memory[read_address_count] : '0;
        burst_length_bytes = s_axis_tdata[22:0];
        burst_length = burst_length_bytes / NUM_PE;
        read_address = s_axis_tdata[63:32] / NUM_PE;
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            read_address_count <= 0;
        end else if (!load_pointer_n) begin
            read_address_count <= read_address;
        end else if (read_address_enable) begin
            read_address_count <= read_address_count + 1;
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn || !load_pointer_n) begin
            read_beat_count <= 0;
        end else if (read_address_enable) begin
            read_beat_count <= read_beat_count + 1;
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            random_seed <= 0;
            current_state <= WMEM_IDLE;
        end else begin
            random_seed <= random_seed + 1;
            current_state <= next_state;
        end
    end

    always_comb begin
        next_state = WMEM_IDLE;
        load_pointer_n = 1'b1;
        s_axis_tready = 1'b0;
        read_address_enable = 1'b0;
        m_axis_tvalid = 1'b0;
        m_axis_tlast = 1'b0;
        case (current_state)
            WMEM_IDLE: begin
                next_state = WMEM_IDLE;
                if (s_axis_tvalid) begin
                    load_pointer_n = 1'b0;
                    next_state = WMEM_READ;
                end
            end
            WMEM_READ: begin
                next_state = WMEM_READ;
                m_axis_tvalid = $urandom(random_seed);
                read_address_enable = m_axis_tvalid && m_axis_tready;

                if ((read_beat_count == burst_length - 2) && read_address_enable) begin
                    s_axis_tready = 1'b1;
                end

                if (read_beat_count == burst_length - 1) begin
                    m_axis_tlast = 1'b1;
                    if (read_address_enable) begin
                        if (s_axis_tvalid) begin
                            load_pointer_n = 1'b0;
                        end else begin
                            next_state = WMEM_IDLE;
                        end
                    end
                end
            end
            default: begin
                next_state = WMEM_IDLE;
            end
        endcase
    end

    initial begin
        if (NUM_PARAMS <= 0) begin
            $fatal(1, "NUM_PARAMS must be positive");
        end
        $readmemh(PARAM_FILE, memory);
    end

    logic unused_s_axis_tlast;
    assign unused_s_axis_tlast = s_axis_tlast;

endmodule

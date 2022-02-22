//-----------------------------------------------------
// Input Generator for Simulation
// Design Name : gen_input
// File Name   : gen_input.sv
//-----------------------------------------------------
`timescale 1ns/1ps

module gen_input # (
        parameter integer NUM_PE    = 8,
        parameter integer ACT_BW    = 8,
        parameter integer NZI_BW    = 16
    )(
    input  logic                     s_axi_aclk,
    input  logic                     s_axi_aresetn,
    // M_AXIS Interface (Input)             
    output logic                     m_axis_tvalid,
    input  logic                     m_axis_tready,
    output logic [NUM_PE*ACT_BW-1:0] m_axis_tdata,
    output logic                     m_axis_tlast
    );

    //-----------------------------------------------------------
    // Useful Functions
    //-----------------------------------------------------------

    // function called clogb2 that returns an integer which has the 
    // value of the ceiling of the log base 2.                      
    function integer clogb2 (input integer bit_depth);              
    begin                                                           
        for(clogb2=0; bit_depth>0; clogb2=clogb2+1)                   
            bit_depth = bit_depth >> 1;                                 
        end                                                           
    endfunction

    //-----------------------
    //-- Logic
    //-----------------------
    logic [NUM_PE*ACT_BW-1:0] mem [];
    longint cnt_rd;
    longint cnt_wait;
    logic re;
    logic cnt_wait_en;
    longint num_mem_entry;

    // State Machine
    typedef enum logic [1:0]   {IG_IDLE = 2'h0,
                                IG_READ = 2'h1,
                                IG_END  = 2'h2} IG_STATE;
    IG_STATE ig_cs, ig_ns;


    //-----------------------
    //-- Memory Packaging
    //----------------------- 
    always_comb begin
        // I/O
        m_axis_tdata = mem[cnt_rd];
    end

    //-----------------------
    //-- Controller
    //-----------------------
    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            cnt_rd <= '0;
            cnt_wait <= '0;
        end else begin
            // Read Beat Counter
            if (re) begin
                cnt_rd <= cnt_rd + 1;
            end
            // Wait Counter
            if (cnt_wait_en) begin
                cnt_wait <= cnt_wait + 1;
            end
        end
    end

    // FSM - Sequencial Logic
    always_ff @ (posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            ig_cs <= IG_IDLE;
        end else begin
            ig_cs <= ig_ns;
        end
    end

    // FSM - Combinational Logic
    always_comb begin
        ig_ns         = IG_IDLE;
        re            = '0;
        m_axis_tvalid = '0;
        m_axis_tlast  = '0;
        cnt_wait_en   = '0;
        case(ig_cs)
            // Initialization
            IG_IDLE: //------------------------------------------------
            begin
                ig_ns = IG_IDLE;
                cnt_wait_en = '1;
                if (cnt_wait > 1000) begin
                    m_axis_tvalid = s_axi_aresetn;
                    if (m_axis_tready) begin
                        ig_ns = IG_READ;
                        re = '1;
                    end
                end
            end
            
            // IDLE
            IG_READ: //------------------------------------------------
            begin
                ig_ns = IG_READ;
                m_axis_tvalid = '1;
                re = m_axis_tready;
                if (re) begin
                    if (cnt_rd == num_mem_entry-1) begin
                        ig_ns = IG_END;
                    end else begin
                        ig_ns = IG_IDLE;
                    end
                end
            end
            
            IG_END: //------------------------------------------------
            begin
                ig_ns = IG_END;
            end
            
            default: //-------------------------------------------
            begin
                ig_ns          = IG_IDLE;
                re             = '0;
                m_axis_tvalid  = '0;
                m_axis_tlast   = '0;
                cnt_wait_en    = '0;
            end
        endcase
    end

    initial begin : read_file
        $readmemh("../../../../../../hdl/tb/edgedrnn_tb_stim.txt", mem);
        num_mem_entry = 0;
        foreach(mem[i]) begin
            num_mem_entry = num_mem_entry + 1;
        end
    end
endmodule

`default_nettype none

/** Portable AXI4-Stream safety assertions used on every external DUT stream. */
module axis_stream_protocol_checker #(
    parameter string INTERFACE_NAME = "axis"
) (
    axis_stream_if.monitor axis
);
    timeunit 1ns;
    timeprecision 1ps;

    // A producer may withdraw or replace a beat only after a handshake.
    property p_stable_while_stalled;
        @(posedge axis.clk) disable iff (!axis.reset_n)
            axis.tvalid && !axis.tready
            |=> axis.tvalid && $stable({axis.tdata, axis.tlast});
    endproperty

    // Unknown payload bits make a bit-exact comparison meaningless.
    property p_known_on_transfer;
        @(posedge axis.clk) disable iff (!axis.reset_n)
            axis.tvalid && axis.tready
            |-> !$isunknown({axis.tdata, axis.tlast});
    endproperty

    assert property (p_stable_while_stalled)
        else $fatal(1, "%s changed a beat while stalled", INTERFACE_NAME);
    assert property (p_known_on_transfer)
        else $fatal(1, "%s transferred X/Z data", INTERFACE_NAME);

endmodule

`default_nettype wire

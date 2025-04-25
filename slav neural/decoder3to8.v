module decoder3to8 (
    input [2:0] in,       // 3-bit input
    input en,             // Enable signal
    output reg [7:0] out  // 8-bit output
);

always @(*) begin
    if (en)
        out = 8'b00000001 << in; // Shift 1 to the left by 'in' positions
    else
        out = 8'b00000000;       // All outputs low when disabled
end

endmodule

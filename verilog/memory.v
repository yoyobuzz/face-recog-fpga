module memory #(parameter BITS=32,DEPTH=90)(
    input clk,
    input r_en,
    input w_en,
    input [17:0] r_add,
    input [17:0] w_add,
    input [BITS-1:0] w_data,
    
    output reg [7:0] r_data
    );
    
    reg [BITS-1:0] mem [0:DEPTH-1];
    
    always@(posedge clk) begin
        if (w_en) begin
            mem[w_add] <= w_data;
        end
        if (r_en) begin
            r_data <= mem[r_add];
        end
    end
endmodule
`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07.04.2024 14:37:04
// Design Name: 
// Module Name: multiplier
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module multiplier(
input [31:0] A,B,
output [31:0] O
    );
    //assign avail=en;
    assign O[31]=A[31]^B[31];
    reg [22:0] P_mant;
    assign O[22:0]=P_mant;
    reg [7:0] P_exp;
    assign O[30:23]=P_exp; 
    
    reg [23:0] A_mant,B_mant;
    reg [47:0] O_mant;
    always@(*)
    begin
    A_mant={1'b1,A[22:0]};
    B_mant={1'b1,B[22:0]};
    end
    always@(*)
    begin
        O_mant=A_mant*B_mant;
        case(O_mant[47])
        0:
        begin
            P_mant={O_mant[45:23]};
        end
        1:
        begin
            P_mant={O_mant[46:24]};
        end
        default:
        begin
            P_mant={O_mant[46:24]};
        end
        endcase
    end
    always@(*)
    begin
        P_exp=A[30:23]+B[30:23]+O_mant[47]-127;
    end
endmodule

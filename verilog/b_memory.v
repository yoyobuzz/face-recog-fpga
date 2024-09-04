`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12.04.2024 18:40:18
// Design Name: 
// Module Name: b_memory
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


module b_memory#(parameter N=1024)(
input clk,reset,
input [31:0] data_in
    );
    
    reg m0_r_en,m0_w_en;
    reg [17:0] wr_pointer_reg,wr_pointer_next,mem_h_reg,mem_h_next;
    reg [4:0] state_next,state_reg;
    wire [31:0] mem_buffer; 
    memory #(.BITS(32),.DEPTH(N)) m0(
    .clk(clk),
    .r_en(m0_r_en),
    .w_en(m0_w_en),
    .r_add(mem_h_reg),
    .w_add(wr_pointer_reg),
    .w_data(data_in),
    .r_data(mem_buffer));
    
    always@(posedge clk)
    begin
        if(reset)
        begin
            state_reg<=0;
            mem_h_reg<=0;
            wr_pointer_reg<=0;
        end
        else
        begin
            state_reg<=state_next;
            mem_h_reg<=mem_h_next;
            wr_pointer_reg<=wr_pointer_next;
        end
        
    end
    
    always@(*)
    begin
        wr_pointer_next=wr_pointer_reg;
        mem_h_next=mem_h_reg;
        state_next=state_reg;
        
        case(state_next)
        0:
        begin
            m0_w_en=1'b1;
            if(wr_pointer_next==N-1)
            begin
                //m0_w_en=1'b0; 
                wr_pointer_next=0;
                state_next=1;
            end
            else
            begin
                wr_pointer_next=wr_pointer_reg+1;
                state_next=0;
            end
            
        end
        1:
        begin
            state_next=1;
        end
        endcase
    end
    

endmodule

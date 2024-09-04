`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10.04.2024 02:16:44
// Design Name: 
// Module Name: seq_mat_mul_fsm
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


module seq_mat_mul_fsm #(parameter N=3,M=2)(
input clk,reset,
input [31:0] data_in
    );
    
    reg [31:0] data_mem [0:N-1];
    reg [31:0] data_weigh_mem [0:N*M-1];
    reg [31:0] data_intermid [0:N*M-1];
    reg [31:0] data_final [0:M-1];
    reg [31:0] added_reg,added_next;
    reg [4:0]state_next,state_reg;
    reg [8:0] wr_pointer_reg,wr_pointer_next;
    reg [8:0] mem_h_reg,weight_h_reg,count_reg,veg_h_reg;
    reg [8:0] mem_h_next,weight_h_next,count_next,veg_h_next;
    wire [31:0] linebuffer;
    multiplier m0(.A(data_mem[mem_h_reg]),.B(data_weigh_mem[weight_h_reg]),.O(linebuffer));
    wire [31:0] lolipop;
    
    
    Addition_Subtraction as0(
    .a_operand(data_intermid[mem_h_reg]),
    .b_operand(added_reg), //Inputs in the format of IEEE-754 Representation.
    .AddBar_Sub(1'b0),	//If Add_Sub is low then Addition else Subtraction.
    .Exception(),
    .result(lolipop)
    
    );
    
    initial
    begin
        state_reg=0;
        wr_pointer_reg=0;
        count_reg=0;
        mem_h_reg=0;
        weight_h_reg=0;
        veg_h_reg=M;
        added_reg=0;
    end
    
    always@(posedge clk)
    begin
        if(reset)
        begin
            state_reg<=0;
            wr_pointer_reg<=0;
            count_reg<=0;
            mem_h_reg<=0;
            weight_h_reg<=0;
            veg_h_reg<=M;
            added_reg<=0;
        end
        else
        begin
            state_reg<=state_next;
            wr_pointer_reg<=wr_pointer_next;
            weight_h_reg<=weight_h_next;
            count_reg<=count_next;
            mem_h_reg<=mem_h_next;
            veg_h_reg<=veg_h_next;
            added_reg<=added_next;
        end
    end
    
    always@(*)
    begin
        wr_pointer_next=wr_pointer_reg;
        state_next=state_reg;
        count_next=count_reg;
        mem_h_next=mem_h_reg;
        weight_h_next=weight_h_reg;
        veg_h_next=veg_h_reg;
        added_next=added_reg;
        case(state_next)
        0:
        begin
            data_mem[wr_pointer_next]=data_in;
            
            if(wr_pointer_next==N-1)
            begin 
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
            data_weigh_mem[wr_pointer_next]=data_in;
            
            if(wr_pointer_next==N*M-1)
            begin
                state_next=2;
                wr_pointer_next=0;
            end
            else
            begin
                wr_pointer_next=wr_pointer_reg+1;
                state_next=1;
            end
        end
        2:
        begin
            
            data_intermid[wr_pointer_next]=linebuffer;
            if(weight_h_next==N*M-1)
            begin
                state_next=3;
                mem_h_next=0;
                count_next=0;
                weight_h_next=0;
                wr_pointer_next=0;
            end
            else
            begin
                state_next=2;
                wr_pointer_next=wr_pointer_reg+1;
                if(count_next==M-1)
                begin
                    mem_h_next=mem_h_reg+1;
                    count_next=0;
                    weight_h_next=weight_h_reg+1;
                
                end
                else
                begin
                    weight_h_next=weight_h_reg+1;
                    count_next=count_reg+1;
                    mem_h_next=mem_h_reg;
                end 
            end
        end
        3:
        begin
            
            added_next=lolipop;
            mem_h_next=mem_h_reg+M;
            //veg_h_next=veg_h_reg+M;
           // state_next=3;
            if(mem_h_reg+M>=M*N)
                begin
                    data_final[count_next]=lolipop;
                    added_next=0;
                    //added_next=0;
                    count_next=count_reg+1;
                    mem_h_next=mem_h_reg-(M*(N-1))+1;
                    //veg_h_next=mem_h_reg-(M*(N-2))+1+M;
                    if(mem_h_reg-(M*(N-2))+1==M)
                    begin
                        state_next=4;
                    end
                    else
                    begin
                        state_next=3;
                    end
                    
                end
            
            
        end
        4:
        begin
            state_next=4;
        end
        endcase
    end
    
     
    
endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Craeate Date: 10.04.2024 02:16:44
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


module seq_mat_mul_fsm_1 #(parameter N=3,M=2)(
input clk,reset,
input [31:0] data_in,
output [0:M-1] maxpo
    );
    
    reg [31:0] data_mem [0:N-1];
    reg [31:0] data_weigh_mem [0:N*M-1];
    reg [31:0] data_coeff_mem[0:M-1];
    reg [31:0] data_intermid [0:N*M-1];
    reg [31:0] data_final [0:M-1];
    reg [31:0] data_final_val [0:M-1];
    reg [31:0] added_reg,added_next;
    reg [4:0] state_next,state_reg;
    reg [17:0] wr_pointer_reg,wr_pointer_next;
    reg [17:0] mem_h_reg,weight_h_reg,count_reg,veg_h_reg;
    reg [17:0] mem_h_next,weight_h_next,count_next,veg_h_next;
    wire [31:0] linebuffer;
    wire [31:0] lolipop,raspberry,blueberry;
    reg [17:0] index_next,index_reg;
    reg [31:0] max_val_reg,max_val_next;
    reg [31:0] val_now;
    reg [31:0] lin_reg,lin1_reg,lin2_reg,lin3_reg,lin4_reg,lin5_reg;
    reg [31:0] yoyo_reg,yoyo1_reg,yoyo2_reg,yoyo3_reg,yoyo4_reg,yoyo5_reg;
    
    integer p;
    initial 
    begin
        weight_h_reg=0;
        mem_h_reg=0;
        wr_pointer_reg=0;
        for(p=0;p<N;p=p+1)
        begin
            data_mem[p]=0;
        end
        
    end
    initial
    begin
        index_reg=0;
        max_val_reg=0;
        state_reg=0;
        wr_pointer_reg=0;
        count_reg=0;
        mem_h_reg=0;
        weight_h_reg=0;
        veg_h_reg=M;
        added_reg=0;
       
    end
    /*always@(*)
    begin
        data_mem[mem_h_reg]=lin_reg;
        data_weigh_mem[weight_h_reg]=lin1_reg;
        data_intermid[mem_h_reg]=lin2_reg;
        data_final[wr_pointer_reg]=lin3_reg;
        data_coeff_mem[wr_pointer_reg]=lin4_reg;
        data_final_val[wr_pointer_reg]=lin5_reg;
    end
    
    always@(*)
    begin
        data_mem[wr_pointer_next]=yoyo_reg;
        data_weigh_mem[wr_pointer_next]=yoyo1_reg;
        data_coeff_mem[wr_pointer_next]=yoyo2_reg;
        data_intermid[wr_pointer_next]=yoyo3_reg;
        data_final[count_next] = yoyo4_reg;
        data_final_val[wr_pointer_next]=yoyo5_reg;
        data_weigh_mem[weight_h_reg]=lin1_reg;
        data_intermid[mem_h_reg]=lin2_reg;
        data_final[wr_pointer_reg]=lin3_reg;
        data_coeff_mem[wr_pointer_reg]=lin4_reg;
        data_final_val[wr_pointer_reg]=lin5_reg;
    end*/
    

    
    
    multiplier m0(.A(data_mem[mem_h_reg]),.B(data_weigh_mem[weight_h_reg]),.O(linebuffer));
    //multiplier m0(.A(lin_reg),.B(lin1_reg),.O(linebuffer));
    Addition_Subtraction as0(
    .a_operand(data_intermid[mem_h_reg]),
    .b_operand(added_reg), //Inputs in the format of IEEE-754 Representation.
    .AddBar_Sub(1'b0),	//If Add_Sub is low then Addition else Subtraction.
    .Exception(),
    .result(lolipop)
    
    );

   /* always@(*)
    begin
        if(state_reg==2)
        begin
            lin_reg=data_intermid[mem_h_reg];
        end
        else
            lin_reg=0;
    end*/    
     Addition_Subtraction as1(
    .a_operand(data_final[wr_pointer_reg]),
    .b_operand(data_coeff_mem[wr_pointer_reg]), //Inputs in the format of IEEE-754 Representation.
    .AddBar_Sub(1'b0),	//If Add_Sub is low then Addition else Subtraction.
    .Exception(),
    .result(raspberry)
    
    );
    
     Addition_Subtraction as2(
    .a_operand(data_final_val[wr_pointer_reg]),
    .b_operand(max_val_reg), //Inputs in the format of IEEE-754 Representation.
    .AddBar_Sub(1'b1),	//If Add_Sub is low then Addition else Subtraction.
    .Exception(),
    .result(blueberry)
    
    );

    
    always@(posedge clk)
    begin
        if(reset)
        begin
            index_reg<=0;
            state_reg<=0;
            wr_pointer_reg<=0;
            count_reg<=0;
            mem_h_reg<=0;
            weight_h_reg<=0;
            veg_h_reg<=M;
            added_reg<=0;
            max_val_reg<=0;
            
        end
        else
        begin
            index_reg<=index_next;
            state_reg<=state_next;
            wr_pointer_reg<=wr_pointer_next;
            weight_h_reg<=weight_h_next;
            count_reg<=count_next;
            mem_h_reg<=mem_h_next;
            veg_h_reg<=veg_h_next;
            added_reg<=added_next;
            max_val_reg<=max_val_next;
           
        end
    end
    
    always@(*)
    begin
        index_next=index_reg;
        wr_pointer_next=wr_pointer_reg;
        state_next=state_reg;
        count_next=count_reg;
        mem_h_next=mem_h_reg;
        weight_h_next=weight_h_reg;
        veg_h_next=veg_h_reg;
        added_next=added_reg;
        max_val_next=max_val_reg;
           
        case(state_next)
        0:
        begin
            data_mem[wr_pointer_next]=data_in;
           // yoyo_reg=data_in;
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
            //yoyo1_reg=data_in;
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
            data_coeff_mem[wr_pointer_next]=data_in;
            //yoyo2_reg=data_in;
            if(wr_pointer_next==M-1)
            begin
                state_next=3;
                wr_pointer_next=0;
               
            end
            else
            begin
                wr_pointer_next=wr_pointer_reg+1;
                state_next=2;
            end
        end
        3:
        begin
            
            data_intermid[wr_pointer_next]=linebuffer;
            //yoyo3_reg=linebuffer;
            if(weight_h_next==N*M-1)
            begin
                state_next=4;
                mem_h_next=0;
                count_next=0;
                weight_h_next=0;
                wr_pointer_next=0;
            end
            else
            begin
                state_next=3;
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
        4:
        begin
            
            added_next=lolipop;
            mem_h_next=mem_h_reg+M;
            veg_h_next=veg_h_reg+M;
           // state_next=3;
            if(mem_h_reg+M>=M*N)
                begin
                    data_final[count_next]=lolipop;
                    //yoyo4_reg=lolipop;
                    added_next=0;
                    //added_next=0;
                    count_next=count_reg+1;
                    mem_h_next=mem_h_reg-(M*(N-1))+1;
                    //veg_h_next=mem_h_reg-(M*(N-2))+1+M;
                    if(mem_h_reg-(M*(N-1))+1==M)
                    begin
                        state_next=5;
                        count_next=0;
                        mem_h_next=0;
                    end
                    else
                    begin
                        state_next=4;
                    end
                    
                end
            
            
        end
        5:
        begin
            //yoyo5_reg=raspberry;
            data_final_val[wr_pointer_next]=raspberry;
            wr_pointer_next=wr_pointer_reg+1;
            if(wr_pointer_reg==M-1)
            begin
                state_next=6;
                wr_pointer_next=0;
            end
            else
                state_next=5;
        end
        6:
        begin
            if(blueberry[31]==1 && wr_pointer_reg!=0)
            begin
                max_val_next=max_val_reg;
                index_next=index_reg;
            end
            else
            begin
                max_val_next=data_final_val[wr_pointer_reg];
                index_next=wr_pointer_reg;
            end
            wr_pointer_next=wr_pointer_reg+1;
            if(wr_pointer_reg==M-1)
                begin
                    state_next=7;
                    wr_pointer_next=0;
                end
            else
                state_next=6;
            
        end
        7:
        begin
            state_next=7;
        end
        default:
        begin
            state_next=7;
//             data_mem[wr_pointer_next]=0;
//            data_final_val[wr_pointer_next]=0;
//            data_final[count_next]=0;
//            data_coeff_mem[wr_pointer_next]=0;
//            data_intermid[wr_pointer_next]=0;
//            data_mem[wr_pointer_next]=0;
//            data_weigh_mem[wr_pointer_next]=0;
        end
        endcase
    end
    reg [0:M-1] max_po_next;
    always@(*)
    begin
        max_po_next=0;
        max_po_next[index_reg]=1;    
    end    
    assign maxpo=max_po_next;
     
    
endmodule

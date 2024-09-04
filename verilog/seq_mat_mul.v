`timescale 1ns / 1ps

module seq_mat_mul#(parameter N=3,M=2)(
input clk,reset,
input [31:0] data_in

    );
    
    reg [31:0] data_mem [0:N-1];
    reg [31:0] data_weigh_mem [0:N*M-1];
    reg [31:0] data_intermid [0:N*M-1];
    //reg [31:0] data_intermid1 [0:N*M-1];
    reg [31:0] data_final [0:M-1];
    reg [8:0] mem_h,weight_h,veg_h,mem1_h;
    reg [8:0] count;
    reg check;
    reg start;
    reg yoyo;
    reg teja;
    reg [8:0] wr_pointer;
    initial
    begin
        teja=0;
        yoyo=0;
        start=0;
        check=0;
        veg_h=M;
        wr_pointer=0;
        mem_h=0;
        weight_h=0;
        count=0;
        mem1_h=0;
    end
    always@(posedge clk)
    begin
        if(reset)
        begin
            teja<=0;
            mem1_h<=0;
            yoyo<=0;
            veg_h<=M;
            wr_pointer<=0;
            check<=0;
            start<=0;
            mem_h<=0;
            weight_h<=0;
            count<=0;
        end
        else
        begin
            if(yoyo==0 && teja==0)
            begin
                if(start==0)
                wr_pointer<=wr_pointer+1;
            
            if(wr_pointer==N && check==0 && start==0)
            begin
                check<=1;
                wr_pointer<=1;
                data_weigh_mem[0]<=data_in;
            end
            else
            begin
                if(check==0 && start==0)
                begin
                    data_mem[wr_pointer]<=data_in;
                end
                else
                if(start==0)
                begin
                    if(wr_pointer<N*M)
                        data_weigh_mem[wr_pointer]<=data_in;
                    else
                    begin
                        start<=1;
                        wr_pointer<=0;
                    end
                        
                end
            end
            end
            
                
            
            
            
        end
    end
    
    wire [31:0] linebuffer;
    
    multiplier m0(.A(data_mem[mem_h]),.B(data_weigh_mem[weight_h]),.O(linebuffer));
    always@(posedge clk)
    begin
        if(start)
        begin
            wr_pointer<=wr_pointer+1;
            data_intermid[wr_pointer]<=linebuffer;
            count<=count+1;
            if(weight_h==N*M)
            begin
                start<=0;
                yoyo<=1;
                count<=0;
                wr_pointer<=0;
                mem_h<=0;
                weight_h<=0;
            end
            else
            begin
                if(count==M-1)
                begin
                    mem_h<=mem_h+1;
                    count<=0;
                    weight_h<=weight_h+1;
                
                end
                else
                begin
                    weight_h<=weight_h+1;
                    count<=count+1;
                    mem_h<=mem_h;
                end 
            end
                      
        end
    end
    
    
    wire [31:0] lolipop;
    
    
    Addition_Subtraction as0(
    .a_operand(data_intermid[mem1_h]),
    .b_operand(data_intermid[veg_h]), //Inputs in the format of IEEE-754 Representation.
    .AddBar_Sub(1'b0),	//If Add_Sub is low then Addition else Subtraction.
    .Exception(),
    .result(lolipop)
    
    );
    
    
    
    always@(posedge clk)
    begin
        if(yoyo)
        begin
            mem1_h<=veg_h;
            veg_h<=veg_h+M;
            if(veg_h+M>=M*N)
                begin
                    mem1_h<=mem1_h-(M*(N-2))+1;
                    veg_h<=mem1_h-(M*(N-2))+1+M;
                    if(mem1_h-(M*(N-2))+1==M)
                    begin
                        yoyo<=0;
                        teja<=1;
                    end
                end
            data_intermid[veg_h]<=lolipop;
        end
    end
    
    
    
endmodule

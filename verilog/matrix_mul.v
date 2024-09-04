`timescale 1ns/1ps



module matrix_mul #(parameter N=4,M=3)(
input [0:2][31:0] x_element ,
input [0:5][31:0] x_weight,
//input [Bits-1:0][0:M-1] x_coeff,
output [0:5][31:0] x_int,
output [0:1][31:0] x_output
    );


wire [31:0] intermid [0:5];

generate
genvar i,j;
begin

    for(i=0;i<N;i=i+1)
    begin
        for(j=0;j<M;j=j+1)
        begin
        multiplier m0(
        .A(x_element[i]),
        .B(x_weight[i*N+j]),
        .O(intermid[i*N+j])
        
        );
        end
    end
end
endgenerate

assign x_int[0]=intermid[0];
assign x_int[1]=intermid[1];
assign x_int[2]=intermid[2];
assign x_int[3]=intermid[3];
assign x_int[4]=intermid[4];
assign x_int[5]=intermid[5];
/*
wire [31:0] sumint [0:N*M];
reg [31:0] sumfinal [0:M-1];
*/
/*
integer p,q;
always@(*)
begin
    
    for(p=0;p<M;p=p+1)
    begin
    sumint=0;
        for(q=0;q<N;q=q+1)
        begin
            //sumint=sumint+intermid[N+q*M];
        end
        
    end
end
*/
/*
generate 
genvar p,q;
for(p=0;p<M;p=p+1)
begin
    for(q=0;q<N;q=q+1)
    begin
        Addition_Subtraction AS0(
        .a_operand(sumint[q+M*p]),
        .b_operand(intermid[q+M*p]), //Inputs in the format of IEEE-754 Representation.
        .AddBar_Sub(1'b0),	//If Add_Sub is low then Addition else Subtraction.
        .Exception(),
        .result(sumint[q+M*p+1]) 
        );
    end
end
endgenerate
integer l;
always@(*)
begin
for(l=0;l<M;l=l+1)
begin
    sumfinal[l]=sumint[N+l*M];
end
   
end
*/
endmodule
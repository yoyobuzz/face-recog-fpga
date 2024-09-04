
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/30/2021 03:02:43 PM
// Design Name: 
// Module Name: terminal_demo
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


module terminal_demo
    #(parameter IMAGE_WIDTH=4,IMAGE_HEIGHT=4)
    (
    input clk, reset_n,
    
    // Receiver port
    
    output rx_empty,    // LED0
    input rx   ,
    output v_sync,h_sync ,
    output [11:0] pixel
    //input [3:0] inbutt,       
    //input rd_mem,
    // Transmitter port
//    input [7:0] w_data, // SW0 -> SW7
//    input wr_uart,      // right push button
//    output tx_full,     // LED1
//    output tx,
    
    // Sseg signals
    
   // output [11:0] leds
    
    
    );
    wire [9:0] v_loc;
    wire [10:0] h_loc;
    wire clk_40;
    integer i;
    reg [7:0] R1,R2;
    reg [11:0] val;
    wire [11:0] out_pixel;
    wire [11:0] leds_next,leds_next_1;
    reg [$clog2(IMAGE_WIDTH*IMAGE_HEIGHT)-1:0] wr_pointer; 
    reg count;
    reg wr_enable;
    wire [11:0] read_val;
    reg [$clog2(IMAGE_WIDTH*IMAGE_HEIGHT)-1:0] counter;
    
    initial 
    begin
        i=0;
        count=0;
        wr_pointer=4'b1111;
        counter<=0;
        wr_enable<=0;
    end
    
   
    
    memory 
    #(.IMAGE_WIDTH(800),.IMAGE_HEIGHT(600))
    m0
    (
        .clk(clk),
        .r_en(1'b1),
        .w_en(count),
        .r_add(h_loc+v_loc*WIDTH),
        .w_add(wr_pointer),
        .w_data(val),
        .r_data(read_val)
    );
    
    
    /*button read_mem(
        .clk(clk),
        .reset_n(reset_n),
        .noisy(rd_mem),
        .debounced(),
        .p_edge(rd_mem_pedge),
        .n_edge(),
        ._edge()
    );*/
        
       
   // assign leds =  read_val;
        
    // UART Driver
    wire [7:0] r_data;
    UART #(.DBIT(8), .SB_TICK(16)) uart_driver(
        .clk(clk),
        .reset_n(reset_n),
        .r_data(r_data),
        .rd_uart(1),
        .rx_empty(rx_empty),
        .rx(rx),
        .w_data(),
        .wr_uart(0),
        .tx_full(),
        .tx(),
        .TIMER_FINAL_VALUE(11'd650) // baud rate = 9600 bps
        
    );
    
    always @(negedge rx_empty)
    begin
        
        begin
            count<=count+1;
            if(count==1)
            begin
                wr_pointer<=wr_pointer +1;
            end
            else
            begin
                wr_pointer<=wr_pointer;
            end
        end
        R1<=r_data;
        R2<=R1;
    end
    
    always@(*)
        val={R2,R1[7:4]};
    
    
    always @(negedge count)
    begin
        if(count==0)
        begin
            wr_enable <= 1;
        end
        else
        begin
            wr_enable<=0;
        end
    end
    
      clk_wiz_0 instance_name
   (
    // Clock out ports
    .clk_out1(clk_40),     // output clk_out1
    // Status and control signals
    .reset(reset_n), // input reset
    .locked(locked),       // output locked
   // Clock in ports
    .clk_in1(clk)      // input clk_in1
);

wire v_disp,h_disp;
reg [11:0] pixel_reg;
disp_sync D0(.clk(clk_40),.rst(reset_n),.v_sync(v_sync),.h_sync(h_sync),.v_disp(v_disp),.h_disp(h_disp),.h_loc(h_loc),.v_loc(v_loc));

always@(*)
begin
if(v_disp && h_disp)
begin
    pixel_reg=read_val;
end
begin
    
end
end
assign pixel=pixel_reg;
endmodule

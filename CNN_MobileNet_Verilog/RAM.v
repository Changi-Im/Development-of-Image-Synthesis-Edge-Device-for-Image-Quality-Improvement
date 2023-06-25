module RAM(qp,qw,dp,dw,write_addressp,read_addressp,write_addressw,read_addressw,we_p,we_w,re_p,re_w,clk);
	parameter picture_size			= 0;
	parameter SIZE_1				= 0;
	parameter SIZE_2				= 0;
	parameter SIZE_4				= 0;
	parameter SIZE_12				= 0;
	parameter SIZE_address_pix		= 0;
	parameter SIZE_address_wei		= 0;

	output	reg	signed	[SIZE_1-1:0] 				qp;       							// read data from RAM to conv_TOP using memorywork
	output 	reg signed 	[SIZE_12-1:0] 				qw;      							// read weight data from RAM to conv_TOP using memorywork
	input 		signed 	[SIZE_1-1:0] 				dp;   								// write data from conv_TOP(pixel after mac caluation) or database_pixel(start pixel) to RAM 
	input 		signed 	[SIZE_12-1:0] 				dw;   								// write weight from database_weight to RAM
	input 				[SIZE_address_pix-1:0]		write_addressp, read_addressp;		// data address
	input 				[SIZE_address_wei-1:0] 		write_addressw, read_addressw;      // weight address
	input 											we_w, re_p, we_p, re_w, clk;        // signal

	reg    signed  [SIZE_1-1:0]    mem     [0:picture_size*picture_size*50-1];          // memory
	reg    signed  [SIZE_12-1:0]   weight  [0:13];									    // max 12 kernel
	
	always @ (posedge clk) begin
		if (we_p) 	mem[write_addressp] 	<= dp;
		if (we_w) 	weight[write_addressw] 	<= dw;
	end
	
	always @ (posedge clk) begin
		if (re_p) 	qp 	<= mem[read_addressp];
		if (re_w) 	qw 	<= weight[read_addressw];
	end
endmodule
module TOP(clk, valid, dp_database, STOP, image_final, image_final_en);

	parameter num_conv  = 1;
	parameter SIZE_1 	= 14;                                                                // bits
	parameter SIZE_2 	= SIZE_1*2;
	parameter SIZE_3 	= SIZE_1*3;
	parameter SIZE_4 	= SIZE_1*4;
	parameter SIZE_5 	= SIZE_1*5;
	parameter SIZE_6 	= SIZE_1*6;
	parameter SIZE_7 	= SIZE_1*7;
	parameter SIZE_8 	= SIZE_1*8;
	parameter SIZE_9 	= SIZE_1*9;
	parameter SIZE_10 	= SIZE_1*10;
	parameter SIZE_11 	= SIZE_1*11;
	parameter SIZE_12 	= SIZE_1*12;
	parameter SIZE_address_pix 	       = 24;
	parameter SIZE_address_wei 	       = 11;
	parameter picture_size 		       = 40;                                                   // picture size 40 X 40
	parameter picture_storage_limit    = 3200;                                              // input picture2 (40x40 + 40x40)
	parameter picture_storage_limit_2  = picture_size*picture_size*14+1;                    // input feature map 2, conv feature map 12
    parameter convolution_size_1by1    = 1;                                                 // PW kernel size
	parameter convolution_size_3by3    = 9;                                                 // DW kernel size
	parameter image2_conv_addr         = picture_size*picture_size*9;
	
	input 							   clk;					                                 // clock
	input          					   valid;						                         // AXI4 valid
	input signed 	[SIZE_1-1:0] 	   dp_database;                                          // pixel data
	output reg 						   STOP;                                                 // finish Program
	output signed  [23:0]              image_final;                                          // final image data
	output reg                         image_final_en;                                       // final image data enable signal	
	
	wire signed 	[SIZE_1-1:0] 		data_p, data_w;                                       // output pixel / weight from RAM to memorywork
	wire 								re_RAM_w, re_RAM_p;                                   // read RAM signal pixel / weight
	wire 			[15:0] 				address;                                              // write pixel / weight from testbech to RAM
	reg 								conv_DW_en, conv_PW_en;                               // conv enblae signal
	wire 								STOP_conv;                                            // conv_TOP STOP
	reg 								STOP_en;                                              // STOP enable signal
	reg 	[4:0] 						TOPlvl;                                               //
	reg 	[5:0] 						lvl;					                              //
	reg 	[6:0] 						slvl;                                                 // output feature map count - 1 
	
	reg 	[SIZE_address_pix-1:0] 		memstartp;                                            // read pixel address starting point 
	reg		[SIZE_address_wei-1:0] 		memstartw;                                            // read weight address starting point    
	wire 	[SIZE_address_wei-1:0]		memstartw_lvl;                                        // weight first address
	reg 	[SIZE_address_pix-1:0] 		memstartzap;                                          // write pixel address starting point
	wire 	[SIZE_address_pix-1:0] 		read_addressp;                                        // read pixel address
	wire 	[SIZE_address_wei-1:0] 		read_addressw;                                        // read weight address
	wire 	[SIZE_address_pix-1:0] 		read_addressp_conv;                                   // read pixel address for conv_TOP

	wire 	[SIZE_address_wei-1:0] 		read_addressw_conv;                                   // write weight address for conv_TOP

	wire 	[SIZE_address_pix-1:0] 		write_addressp;                                       // write pixel address 
	wire 	[SIZE_address_wei-1:0] 		write_addressw;                                       // write weight address 
	wire 	[SIZE_address_pix-1:0] 		write_addressp_memorywork;                            // write pixel address from memorywork
	wire 	[SIZE_address_pix-1:0] 		write_addressp_conv;                                  // write pixel address for conv_TOP

	wire 								we_p,we_w;                                            // write pixel / weight enable signal
	wire 								re_p,re_w;                                            // read pixel / weight enable signal
	wire 								we_p_memorywork;                                      // wrire pixel signal from memorywork
	wire 								we_conv,re_conv,re_wb_conv;                           // write, read pixel / read weight enable signal for conv_TOP

	wire signed [SIZE_1-1:0] 			qp;                                                    // pixel data
	wire signed [SIZE_12-1:0] 			qw;                                                    // weight data
	wire signed [SIZE_1-1:0] 			dp;                                                    // pixel data
	wire signed [SIZE_12-1:0] 			dw;                                                    // weight data
	wire signed [SIZE_1-1:0] 			dp_conv;                                               // mac output from conv_TOP
	wire signed [SIZE_1-1:0] 			dp_memorywork;                                         // output data from memorywork
	wire 		[1:0] 					prov;                                                  // for zero padding
	wire 		[14:0] 					i_conv;                                                // feature map index
	wire signed [SIZE_2-2:0] 			Y1_DW, Y1_PW;                                          // conv result
	wire signed [SIZE_1-1:0] 			w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12;                // weight
	wire signed [SIZE_1-1:0] 			p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;     // pixel
	wire 								go_conv_DW, go_conv_PW;                                // conv enable signal
	wire 								go_conv_TOP;                                           // conv block enable
	wire 		[6:0] 					step;                                                  // conv precedure
	reg 								nextstep;                                              // step + 1
	reg 		[6:0]					matrix;                                                // feature size
	wire 		[12:0] 					matrix2;                                               // feature size * feature size
	reg 		[6:0] 					o_f;				                                   // output feature map count - 1
	reg 		[6:0] 					i_f;				                                   // input feature map count - 1
    reg                                 de_in_en;                                              // decoder / incoder enabel signal
    reg                                 fu_en;                                                 // fusion layer enable signal
    reg                                 rst_n;                                                 // reset_not
    reg                                 we_database;                                          // write signal
    reg         [15:0]                  address_p_database;                                   // write pixel addres from testbench to RAM
    reg                                 GO;                                                   // start CNN
    
    database_pixel #(SIZE_1) database_pixel(
		.clk(clk),.datata(data_p),.re(re_RAM_p),.address(address),.we(we_database),.dp(dp_database),.address_p(address_p_database));
	
    database_weight #(SIZE_1) database_weight(
		.clk(clk),.datata(data_w),.re(re_RAM_w),.address(address));	
		
    RAM #(picture_size,SIZE_1,SIZE_2,SIZE_4,SIZE_12,SIZE_address_pix,SIZE_address_wei) memory(
		qp,qw,dp,dw,write_addressp,read_addressp,write_addressw,read_addressw,we_p,we_w,re_p,re_w,clk);

    memorywork #(num_conv,picture_size,convolution_size_1by1,convolution_size_3by3,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_10,SIZE_11,SIZE_12,SIZE_address_pix,SIZE_address_wei) block(
		.clk(clk),.we_p(we_p_memorywork),.we_w(we_w), .re_RAM_w(re_RAM_w), .re_RAM_p(re_RAM_p),.addrp(write_addressp_memorywork),.addrw(write_addressw),.dp(dp_memorywork),.dw(dw),.step_out(step),.nextstep(nextstep),.data_p(data_p),.data_w(data_w),.address(address),.GO(GO), .i_f(i_f));
		
    border border(
		.clk (clk), .go (conv_DW_en), .i (i_conv), .matrix (matrix), .prov (prov));
		
    conv_DW #(SIZE_1) conv1(
		clk,Y1_DW,prov,matrix,matrix2,i_conv,p1, p2, p3, p4, p5, p6, p7, p8, p9, w1,w2,w3,w4,w5,w6,w7,w8,w9, go_conv_DW);
	conv_PW #(SIZE_1) conv2(
	   clk, Y1_PW, matrix, matrix2, i_conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, w8, w6, w4, w2, w1, w3, w5, w7, w9, w10, w11, w12, i_f, go_conv_PW);
		
    conv_TOP #(num_conv,SIZE_1,SIZE_2,SIZE_3,SIZE_4,SIZE_5,SIZE_6,SIZE_7,SIZE_8,SIZE_9,SIZE_10,SIZE_11,SIZE_12,SIZE_address_pix,SIZE_address_wei) conv(
		clk,conv_DW_en,conv_PW_en,STOP_conv,memstartp,memstartw_lvl,memstartzap,read_addressp_conv,write_addressp_conv,read_addressw_conv,we_conv,re_wb_conv,re_conv,qp,qw,dp_conv,prov,matrix,matrix2,i_conv,lvl,slvl,Y1_DW,Y1_PW,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,go_conv_TOP,i_f, rst_n, step, de_in_en,fu_en, image_final,image_final_en);

	initial lvl 		       = 0;		
	initial slvl 		       = 0;
	initial memstartw 	       = 0;
    initial conv_DW_en 	       = 0;
	initial	conv_PW_en         = 0;
	initial address_p_database = -1;
	initial GO                 = 1;
	
	always @(posedge clk) begin
		if((valid==1)&(GO==1)) begin				                   // start CNN
		    GO                 = 1;
			STOP 		       = 0;
			nextstep 	       = 1;
			matrix		       = picture_size;
			rst_n              = 1;
			we_database        = 1;
			address_p_database = address_p_database + 1'b1;
			if(address_p_database==3199) begin
			     GO = 0; 
			end
		end	else begin
			nextstep	= 0;
			rst_n       = 0;
			we_database = 0;
		end
		
		if(STOP==0) begin
			if((TOPlvl==1)&&(step==4)) begin			// DW1, image1
				memstartp 	= picture_storage_limit;		
				memstartzap = picture_storage_limit_2;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 0;
				i_f 		= 0;
				matrix 		= picture_size;
			end	
			if((TOPlvl==2)&&(step==4)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==2)&&(step==6)) begin			// PW1, image1
				memstartp   = picture_storage_limit_2;
				memstartzap = picture_storage_limit;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 0;
				matrix      = picture_size;
				de_in_en    = 1;
			end	
			if((TOPlvl==3)&&(step==6)) begin
				nextstep    = 1;
			end
			if((TOPlvl==3)&&(step==8))begin				// DW2, image1
				memstartp   = picture_storage_limit;
				memstartzap = picture_storage_limit_2;
				conv_DW_en  = 1;
				o_f         = 2;
				i_f         = 2;
				matrix      = picture_size;
			end	
			if((TOPlvl==4)&&(step==8)) begin
				nextstep    = 1;
			end
			if((TOPlvl==4)&&(step==10)) begin			// PW2, image1
				memstartp   = picture_storage_limit_2;
				memstartzap = picture_storage_limit;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 2;
				matrix      = picture_size;
			end
			if((TOPlvl==5)&&(step==10)) begin
				nextstep    = 1;
			end
			if((TOPlvl==5)&&(step==12)) begin			// DW3, image1
				memstartp   = picture_storage_limit;
				memstartzap = picture_storage_limit_2;
				conv_DW_en  = 1;
				o_f         = 5;
				i_f         = 5;
				matrix      = picture_size;
			end	
			if((TOPlvl==6)&&(step==12)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==6)&&(step==14)) begin			// PW3, image1
				memstartp   = picture_storage_limit_2;
				memstartzap = picture_storage_limit;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 5;
				matrix      = picture_size;
			end
			if((TOPlvl==7)&&(step==14)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==7)&&(step==16)) begin			// DW4, image1
				memstartp   = picture_storage_limit;
				memstartzap = picture_storage_limit_2;
				conv_DW_en  = 1;
				o_f         = 8;
				i_f         = 8;
				matrix      = picture_size;
			end
			if((TOPlvl==8)&&(step==16)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==8)&&(step==18)) begin			// PW4, image1
				memstartp   = picture_storage_limit_2;
				memstartzap = picture_storage_limit;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 8;
				matrix      = picture_size;
			end
			if((TOPlvl==9)&&(step==18)) begin
				nextstep    = 1;
			end
			
			if((TOPlvl==9)&&(step==20)) begin			// DW1, image2
				memstartp 	= (picture_storage_limit + image2_conv_addr)*2 + matrix2 ;		
				memstartzap = picture_storage_limit_2*2 + image2_conv_addr - 1;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 0;
				i_f 		= 0;
				matrix 		= picture_size;
			end	
			if((TOPlvl==10)&&(step==20)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==10)&&(step==22)) begin			// PW1, image2
				memstartp   = picture_storage_limit_2*2 + image2_conv_addr - 1;
				memstartzap = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 0;
				matrix      = picture_size;
			end	
			if((TOPlvl==11)&&(step==22)) begin
				nextstep    = 1;
			end
			if((TOPlvl==11)&&(step==24))begin				// DW2, image2
				memstartp   = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				memstartzap = picture_storage_limit_2*2 + image2_conv_addr - 1;
				conv_DW_en  = 1;
				o_f         = 2;
				i_f         = 2;
				matrix      = picture_size;
			end	
			if((TOPlvl==12)&&(step==24)) begin
				nextstep    = 1;
			end
			if((TOPlvl==12)&&(step==26)) begin			// PW2, image2
				memstartp   = picture_storage_limit_2*2 + image2_conv_addr - 1;
				memstartzap = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 2;
				matrix      = picture_size;
			end
			if((TOPlvl==13)&&(step==26)) begin
				nextstep    = 1;
			end
			if((TOPlvl==13)&&(step==28)) begin			// DW3, image2
				memstartp   = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				memstartzap = picture_storage_limit_2*2 + image2_conv_addr - 1;
				conv_DW_en  = 1;
				o_f         = 5;
				i_f         = 5;
				matrix      = picture_size;
			end	
			if((TOPlvl==14)&&(step==28)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==14)&&(step==30)) begin			// PW3, image2
				memstartp   = picture_storage_limit_2*2 + image2_conv_addr - 1;
				memstartzap = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 5;
				matrix      = picture_size;
			end
			if((TOPlvl==15)&&(step==30)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==15)&&(step==32)) begin			// DW4, image2
				memstartp   = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				memstartzap = picture_storage_limit_2*2 + image2_conv_addr - 1;
				conv_DW_en  = 1;
				o_f         = 8;
				i_f         = 8;
				matrix      = picture_size;
			end
			if((TOPlvl==16)&&(step==32)) begin
				nextstep    = 1;
			end
			if ((TOPlvl==16)&&(step==34)) begin			// PW4, image2
				memstartp   = picture_storage_limit_2*2 + image2_conv_addr - 1;
				memstartzap = (picture_storage_limit + image2_conv_addr)*2 + matrix2;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 8;
				matrix      = picture_size;
			end
			if((TOPlvl==17)&&(step==34)) begin
				nextstep    = 1;
			end
			
			if ((TOPlvl==17)&&(step==36)) begin			// fusion layer
				memstartp   = picture_storage_limit;
				memstartzap = 0;
				conv_PW_en  = 1;
				o_f         = 11;
				i_f         = 1;
				matrix      = picture_size;
				fu_en       = 1;
			end
			if((TOPlvl==18)&&(step==36)) begin
				nextstep    = 1;
				fu_en       = 0;
			end
			
			if((TOPlvl==18)&&(step==38)) begin			// DW5, decoder
				memstartp 	= 0;		
				memstartzap = picture_storage_limit_2 + 309;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 11;
				i_f 		= 11;
				matrix 		= picture_size;
			end	
			if((TOPlvl==19)&&(step==38)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==19)&&(step==40)) begin			// PW5, decoder
				memstartp   = picture_storage_limit_2 + 309;
				memstartzap = picture_storage_limit + 309;
				conv_PW_en  = 1;
				o_f         = 11;
				i_f         = 11;
				matrix      = picture_size;
				de_in_en    = 0;
			end	
			if((TOPlvl==20)&&(step==40)) begin
				nextstep    = 1;
			end
			if((TOPlvl==20)&&(step==42)) begin			// DW6, decoder
				memstartp 	= picture_storage_limit + 309;		
				memstartzap = picture_storage_limit_2 + 309;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 11;
				i_f 		= 11;
				matrix 		= picture_size;
			end	
			if((TOPlvl==21)&&(step==42)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==21)&&(step==44)) begin			// PW6, decoder
				memstartp   = picture_storage_limit_2 + 309;
				memstartzap = picture_storage_limit + 309;
				conv_PW_en  = 1;
				o_f         = 5;
				i_f         = 11;
				matrix      = picture_size;
			end	
			if((TOPlvl==22)&&(step==44)) begin
				nextstep    = 1;
			end
			if((TOPlvl==22)&&(step==46)) begin			// DW7, decoder
				memstartp 	= picture_storage_limit + 309;		
				memstartzap = picture_storage_limit_2 + 309;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 5;
				i_f 		= 5;
				matrix 		= picture_size;
			end	
			if((TOPlvl==23)&&(step==46)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==23)&&(step==48)) begin			// PW7, decoder
				memstartp   = picture_storage_limit_2 + 309;
				memstartzap = picture_storage_limit + 309;
				conv_PW_en  = 1;
				o_f         = 2;
				i_f         = 5;
				matrix      = picture_size;
			end	
			if((TOPlvl==24)&&(step==48)) begin
				nextstep    = 1;
			end
			if((TOPlvl==24)&&(step==50)) begin			// DW8, decoder
				memstartp 	= picture_storage_limit + 309;		
				memstartzap = picture_storage_limit_2 + 309;
				conv_DW_en 	= 1;
				conv_PW_en  = 0;
				o_f	 		= 2;
				i_f 		= 2;
				matrix 		= picture_size;
			end	
			if((TOPlvl==25)&&(step==50)) begin			
				nextstep    = 1;
			end
			if((TOPlvl==25)&&(step==52)) begin			// PW8, decoder
				memstartp   = picture_storage_limit_2 + 309;
				memstartzap = picture_storage_limit + 309;
				conv_PW_en  = 1;
				o_f         = 0;
				i_f         = 2;
				matrix      = picture_size;
				image_final_en=1;
			end	
			if((TOPlvl==26)&&(step==52)) begin
				nextstep    = 1;
				image_final_en=0;
			end
			
			if((TOPlvl==26)&&(step==53)) begin			
				memstartp  = picture_storage_limit_2 + 309;
				STOP_en    = 1;
			end			
			if(STOP_en==1) begin		 // finish program
				STOP_en  = 0;
				STOP     = 1;
			end
			
			if((STOP_conv)&&(conv_DW_en==1)) begin       // conv DW ends once
				conv_DW_en = 0;
			end
			if((STOP_conv)&&(conv_PW_en == 1)) begin     // conv PW ends once
				conv_PW_en = 0;
			end
		end
	end
	
	
	
	always @(negedge STOP_conv or posedge GO) begin       // channel count
		if(GO) begin									
			lvl      = 0;
			slvl     = 0;
			TOPlvl   = 1;
		end	else begin
            if(o_f!=(1 + (slvl*1)) - 1) begin
                slvl = slvl+1; 
            end else begin
                lvl  = lvl+1; 
                slvl = 0; 
            end 
			
			if(lvl== 1) begin                         
				lvl = 0;
				TOPlvl = TOPlvl+1'b1;
			end
		end
	end

	assign memstartw_lvl   = memstartw + slvl;
	
	assign re_p            = ((conv_DW_en==1)||(conv_PW_en==1))?re_conv:0;
	assign re_w            = ((conv_DW_en==1)||(conv_PW_en==1))?re_wb_conv:0;
	assign read_addressp   = ((conv_DW_en==1)||(conv_PW_en==1))?read_addressp_conv:0;
	assign we_p            = ((step==1)||(step==2))?we_p_memorywork:(((conv_DW_en==1)||(conv_PW_en==1))?we_conv:0);
	assign dp              = ((step==1||(step==2)))?dp_memorywork:(((conv_DW_en==1)||(conv_PW_en==1))?dp_conv:0);
	assign write_addressp  = ((step==1)||(step==2))?write_addressp_memorywork:(((conv_DW_en==1)||(conv_PW_en==1))?write_addressp_conv:0);
	assign read_addressw   = ((conv_DW_en==1)||(conv_PW_en==1))?read_addressw_conv:0;

	assign matrix2         = matrix*matrix;

	assign go_conv_DW      = (conv_DW_en==1)?go_conv_TOP:0; 
    assign go_conv_PW      = (conv_PW_en==1)?go_conv_TOP:0;
endmodule
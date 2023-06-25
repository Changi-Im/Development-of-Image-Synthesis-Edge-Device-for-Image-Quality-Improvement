module conv_TOP(clk,conv_DW_en,conv_PW_en,STOP,memstartp,memstartw,memstartzap,read_addressp,write_addressp,read_addressw,we,re_wb,re,qp,qw,dp,prov,matrix,matrix2,i_2,lvl,slvl,Y1_DW,Y1_PW,w15,w14,w16,w13,w17,w12,w18,w11,w19, w20, w21, w22,p1,p2,p3,p8,p7,p4,p5,p9,p6,p10, p11, p12, go,i_f, rst_n, step,de_in_en,fu_en,image_final,image_final_en);

	parameter num_conv = 0;
	parameter SIZE_1 	= 0;
	parameter SIZE_2 	= 0;
	parameter SIZE_3 	= 0;
	parameter SIZE_4 	= 0;
	parameter SIZE_5 	= 0;
	parameter SIZE_6 	= 0;
	parameter SIZE_7 	= 0;
	parameter SIZE_8 	= 0;
	parameter SIZE_9 	= 0;
	parameter SIZE_10 	= 0;
	parameter SIZE_11 	= 0;
	parameter SIZE_12 	= 0;
	parameter SIZE_address_pix   = 0;
	parameter SIZE_address_wei   = 0;

	input 											clk, conv_DW_en, conv_PW_en;				  	            // clock, conv_TOP enable	
	input 				[1:0] 						prov;											            // edge, padding
	input 				[6:0] 						matrix;											            // feature map line size
	input 				[12:0] 						matrix2;										            // matrix * matrix   
	input 				[SIZE_address_pix-1:0] 		memstartp; 										            // image data first address from RAM
	input 				[SIZE_address_wei-1:0] 		memstartw;										            // weight first address
	input 				[SIZE_address_pix-1:0] 		memstartzap;            	      				            // write image data first address														
	input 				[5:0] 						lvl;											            // ??????????? The number of input feature maps for each layer
	input 				[6:0] 						slvl;											            // ???????????
	output 	reg 		[SIZE_address_pix-1:0] 		read_addressp;									            // input pixel data address
	output 	reg 		[SIZE_address_wei-1:0] 		read_addressw;									            // weight block address
	output 	reg 		[SIZE_address_pix-1:0] 		write_addressp;									            // wirte address for output value
	output 	reg										we, re, re_wb;									            // write result in RAM, request image data from RAM , request weight from RAM
	input 		signed 	[SIZE_1-1:0] 				qp;												            // image data from RAM
	input 		signed 	[SIZE_12-1:0] 				qw;												            // weight data from RAM
	output 		signed	[SIZE_1-1:0] 				dp;												            // write image data in RAM
	output 	reg 									STOP;									                    // conv_TOP STOP
	output 				[14:0] 						i_2;									       	            // feature map index
	input 		signed 	[SIZE_1+SIZE_1-2:0] 		Y1_DW, Y1_PW;										        // conv return
	output	reg	signed	[SIZE_1-1:0] 				w15, w14, w16, w13, w17, w12, w18, w11, w19, w20, w21, w22; // weight
	output 	reg signed 	[SIZE_1-1:0]				p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;		    // pixel
	output 	reg 									go;										                    // conv block enable
	input 				[4:0] 						i_f;									                    // input feature map count - 1
	input                                           rst_n;                                                      // reset_not
	input 		        [6:0] 					    step;                                                       // conv order
	input                                           de_in_en;                                                   // incoder / decoder enable signal
	input                                           fu_en;                                                      // fusion layer enable signal
	output reg signed  [23:0]                       image_final;                                                // final result
	input                                           image_final_en;                                             // final result enable signal
	
	reg signed [SIZE_1-1:0] res_out_1;             // result

	reg signed [SIZE_1-1:0] buff0 [2:0];
	reg signed [SIZE_1-1:0] buff1 [2:0];
	reg signed [SIZE_1-1:0] buff2 [2:0];

	reg        [5:0] 	     marker;               // procedure
	reg 		             pixel_weight_en;	   // pixel_weight_en
	reg        [14:0] 	     i;                    // pexel index / read address for PW
    reg        [14:0] 	     j;                    // write pixel address for PW
    reg        [14:0] 	     k;                    // read fusion / write DW / write decoder
    reg        [14:0] 	     h;                    // read pixel address for DW
    reg        [14:0] 	     cnt;
    reg                      tmp;
    
	initial pixel_weight_en = 0;
	initial marker          = 0;
    initial tmp             = 1;
    initial j               = 0;
    initial k               = 0;
    initial h               = 0;
    initial cnt             = -1;

	always @(posedge clk) begin
		if((conv_DW_en==1) && (tmp == conv_DW_en)) begin				// enable convolution(Depth Wise)
    		if(pixel_weight_en==0) begin
			   case(marker)
					0 : begin 											// request weight from RAM
						re_wb = 1; 
						read_addressw = memstartw;
					end
					1 : begin end
					2 : begin											// write weight from RAM
						w11 = qw[SIZE_1-1:0]; 
						w12 = qw[SIZE_2-1:SIZE_1]; 
						w13 = qw[SIZE_3-1:SIZE_2]; 
						w14 = qw[SIZE_4-1:SIZE_3]; 
						w15 = qw[SIZE_5-1:SIZE_4]; 
						w16 = qw[SIZE_6-1:SIZE_5]; 
						w17 = qw[SIZE_7-1:SIZE_6]; 
						w18 = qw[SIZE_8-1:SIZE_7]; 
						w19 = qw[SIZE_9-1:SIZE_8]; 
					end
					3 : begin 											// initialization
						pixel_weight_en = 1; 
						re_wb 			= 0; 
						marker			= -1; 
					end
				endcase
				marker = marker + 1;
			end else begin					                            // request image data from RAM
				re=1;
				case(marker)
					0 : begin		
						read_addressp	= h + memstartp;                // read first row
						if(step==4) begin
					        read_addressp	= h + 0;                    // image1 start point
						end
						if(step==20) begin
					        read_addressp	= h + matrix*matrix;        // image2 start point
						end 
						if((i-1)<matrix2-matrix) begin		  
							buff2[2] = qp[SIZE_1-1:0];
						end else begin
							buff2[2] = 0;
						end
						
						if(i>=2) begin						            // conv calculation enable
							go = 1;
						end
						
						p1 = buff1[1];  // center
						p2 = buff1[2];  // right
						p3 = buff1[0];  // left
						p8 = buff2[0];  // downright
						p7 = buff0[2];  // up
						p4 = buff2[1];  // downleft 
						p5 = buff0[1];  // upright
						p9 = buff2[2];  // upleft
						p6 = buff0[0];  // down 			
					end
					1 : begin		
						if(i>=matrix-1) begin
							read_addressp = h - matrix + memstartp;       // read second row
							
							if(step==4) begin
					           read_addressp	= h - matrix + 0;         // image1 start point
						    end
						    
						    if(step==20) begin                            // image2 start point
					           read_addressp	= h - matrix + matrix*matrix;
						    end 
						end
					
						go = 0;
											
						buff2[0] = buff2[1];
						buff1[0] = buff1[1];
						buff0[0] = buff0[1];
						buff2[1] = buff2[2];
						buff1[1] = buff1[2];
						buff0[1] = buff0[2];
					end
					2 : begin    
						if(i<matrix2-matrix) begin			               // read third row
							read_addressp = h + matrix + memstartp;                     
							
							if(step==4) begin                              // image1 start point
					           read_addressp	= h + matrix + 0;         
						    end
						    
						    if(step==20) begin                             // image2 start point
					           read_addressp	= h + matrix + matrix*matrix;
						    end 
							
						end
						buff1[2] = qp[SIZE_1-1:0];
						
						if(i>=2) begin                                      // write data to RAM
							write_addressp = memstartzap + k - 2;	   															
							res_out_1 = {Y1_DW[SIZE_1+SIZE_1-2], Y1_DW[SIZE_1+SIZE_1-5:SIZE_1-3]};  // bit slicing
                            /*if(res_out_1<0) begin				         // not use relu function at DW
                                res_out_1 = 0; 
                            end*/
                            we=1;
						end
					end
					3 : begin		
						if(i>=matrix-1) begin
							buff0[2] = qp[SIZE_1-1:0];
						end	else begin 
							buff0[2] = 0;
						end
						we=0;
					end
				endcase
				
				if(marker!=3) begin			                                // marker ++
					marker = marker + 1; 
				end else begin 				                                // marker initialization, i ++
					marker = 0; 
					if(i<matrix2+1) begin
						i = i + 1;
						if(i<=matrix*matrix)begin                           // read address count
						  h = h + 1;
						end
						if(i>=2) begin                                      // write address count
						  k = k + 1;
						end
					end else begin 			                                // STOP conv
						STOP = 1; 
					end
				end
			end
		end else if((conv_PW_en==1) && (tmp == ~conv_PW_en) && ((step == 6)||(step == 22))) begin     // enable convolution
			if(pixel_weight_en==0) begin
			   case(marker)
					0 : begin 				                               // request weight from RAM
						re_wb = 1; 
						read_addressw = memstartw;      
					end
					1 : begin end
					2 : begin				                               // write weight from RAM
						w11 = qw[SIZE_1-1:0]; 
						w18 = qw[SIZE_2-1:SIZE_1]; 
						w12 = qw[SIZE_3-1:SIZE_2];
					end
					3 : begin 				                               // initialization
						pixel_weight_en = 1; 
						re_wb 			= 0; 
						marker			= -1; 
					end
					default: $display("Check pixel_weight_en");
				endcase
				marker = marker + 1;
			end else begin					                               // request image data from RAM
				re=1;
				case(marker)
					0 : begin		
						read_addressp	= i + memstartp - 1; 					
						if(i>=2) begin						               // conv calculation enable
							go = 1;
						end
						p1 = buff1[1];	
					end
					1 : begin		
						go = 0;
						buff1[1] = buff1[2];
					end
					2 : begin
						buff1[2] = qp[SIZE_1-1:0];
						if(i>=2) begin
							write_addressp = memstartzap + j - 1;
							res_out_1 = {Y1_PW[SIZE_1+SIZE_1-2], Y1_PW[SIZE_1+SIZE_1-5:SIZE_1-3]};   // bit slicing                                       
                            if(res_out_1<0) begin				           // Lelu func										
                                res_out_1 = 0; 
                            end
                            we=1;
						end
					end
					3: begin		
						we=0;
					end						
				endcase
				
				if(marker!=3) begin			                                // marker ++
					marker = marker + 1; 
				end else begin 				                                // marker initialization, i ++
					marker = 0; 
					if(i<matrix2+1) begin                              
						i = i + 1;                                          // read address count
						if(i>=2) begin                                      
						  j = j + 1;                                        // write address count
						end 
					end else begin 			                                // STOP conv
						STOP = 1; 
					end
				end
			end
		end else if((conv_PW_en==1) && (tmp == ~conv_PW_en) && ((step != 6)||(step != 22))) begin // enable convolution
			if(pixel_weight_en==0) begin
			   case(marker)
					0 : begin 				                               // request weight from RAM
						re_wb = 1; 
						read_addressw = memstartw;
					end
					1 : begin end
					2 : begin				                               // write weight from RAM
						w11 = qw[SIZE_1-1:0]; 
						w12 = qw[SIZE_2-1:SIZE_1]; 
						w13 = qw[SIZE_3-1:SIZE_2]; 
						w14 = qw[SIZE_4-1:SIZE_3]; 
						w15 = qw[SIZE_5-1:SIZE_4]; 
						w16 = qw[SIZE_6-1:SIZE_5]; 
						w17 = qw[SIZE_7-1:SIZE_6]; 
						w18 = qw[SIZE_8-1:SIZE_7]; 
						w19 = qw[SIZE_9-1:SIZE_8];
						w20 = qw[SIZE_10-1:SIZE_9];
						w21 = qw[SIZE_11-1:SIZE_10];
						w22 = qw[SIZE_12-1:SIZE_11];
					    if(step==36) begin
					      w11 = 1;
						  w12 = 1;
					    end
					end
					3 : begin 				                              // initialization
						pixel_weight_en = 1; 
						re_wb 			= 0; 
						marker			= -1; 
					end
				endcase
				marker = marker + 1;
			end else begin					                              // request image data from RAM
				re=1;
				case(marker)                                              // address takes 2clk, so the address is assigned 2clk in advance
					0 : begin
					    we=0;
					    read_addressp    = i + memstartp - 1;
					    if(step == 36) begin                               // for fusion layer
					       read_addressp = k + memstartp;
					    end
					end    
					1 : begin
					   read_addressp	 = i + matrix2*1+memstartp - 1; 
					   if(step == 36) begin                                // for fusion layer
					       read_addressp = k + memstartp + 33600;
					   end
					end	
					2 : begin
						read_addressp 	= i + matrix2*2+memstartp - 1; 
						p1 	            = qp[SIZE_1-1:0];
					end
					3 : begin			
						read_addressp	= i + matrix2*3+memstartp - 1; 
						p2              = qp[SIZE_1-1:0];
					end
					4 : begin
						read_addressp 	= i + matrix2*4+memstartp - 1;
						p3 				= qp[SIZE_1-1:0];
					end
					5 : begin
						read_addressp	= i + matrix2*5+memstartp - 1; 
						p8 				= qp[SIZE_1-1:0];
					end
					6 : begin
						read_addressp	= i + matrix2*6+memstartp - 1; 
						p7 				= qp[SIZE_1-1:0];
					end
					7 : begin
						read_addressp	= i + matrix2*7+memstartp - 1; 
						p4 				= qp[SIZE_1-1:0];
					end
					8 : begin
					   read_addressp	= i + matrix2*8+memstartp - 1; 
						p5 				= qp[SIZE_1-1:0];
					end
					9 : begin
					   read_addressp	= i + matrix2*9+memstartp - 1;
					    p9              = qp[SIZE_1-1:0];		
					    		
					end
					10 : begin
					   read_addressp	= i + matrix2*10+memstartp - 1; 
						p6 				= qp[SIZE_1-1:0];
					end
					11 : begin
					   read_addressp	= i + matrix2*11+memstartp - 1;
					   p10             = qp[SIZE_1-1:0];
					end
					12 : begin
					   p11              = qp[SIZE_1-1:0];
					   if(i>=0) begin						                // conv calculation enable
							go = 1;
						end	
					end
					13 : begin
					   p12              = qp[SIZE_1-1:0];
					end
					14 : begin end
					15 : begin
					   go = 0;
					   if((i>=0) && (i<matrix*matrix)) begin
                            if(de_in_en == 1) begin				            // incoder write address
                                write_addressp = memstartzap + j;		
                            end else begin                                  // decoder write address
                                write_addressp = memstartzap + k;
                            end
                            k = k + 1;
                            j = j + 1;
                            res_out_1 = {Y1_PW[SIZE_1+SIZE_1-2], Y1_PW[SIZE_1+SIZE_1-5:SIZE_1-3]};  // bit slicing
                            if(fu_en == 1) begin                        // fusion layer result
                               res_out_1 = Y1_PW[SIZE_1-1:0];
                            end
                            if(image_final_en==1) begin                 // final result
                                if(res_out_1[SIZE_1-2] == 1) begin
                                    res_out_1[SIZE_1-3:SIZE_1-10] = 8'b11111111;
                                end
                               image_final = {res_out_1[SIZE_1-3:SIZE_1-10],res_out_1[SIZE_1-3:SIZE_1-10],res_out_1[SIZE_1-3:SIZE_1-10]};
                            end
                            if(res_out_1<0) begin						// Lelu func
                                res_out_1 = 0; 
                            end
                            we=1;
						end
					end
				endcase

				if(marker!=15) begin			                            // marker ++
					marker = marker + 1; 
				end else begin 				                                // marker initialization, i ++
					marker = 0; 
					if(i<matrix2+1) begin
						i = i + 1;
						if(j==matrix*matrix*12)begin                        // to accumulate results
			              j = 0;
			           end 
					end else begin 			                                // STOP conv
						STOP = 1; 
					end
				end
			end
		end else if((tmp == conv_PW_en) || (tmp == ~conv_DW_en)||(rst_n == 1))begin	 // first initialization
			i				= 0;
			pixel_weight_en	= 0;
			STOP			= 0;
			re				= 0;
			go				= 0;
			marker			= 0;
			tmp             = ~tmp;
		end 
	end
	
	always @(step) begin
         k = 0;
         h = 0;
	end
	
	assign i_2 	= i - 2;
	assign dp 	= {res_out_1};
endmodule

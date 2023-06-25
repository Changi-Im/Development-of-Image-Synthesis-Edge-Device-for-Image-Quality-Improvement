module memorywork(clk, data_p, data_w, address, we_p, we_w, re_RAM_w, re_RAM_p, nextstep, dp, dw, addrp, addrw, step_out, GO, i_f);
	parameter num_conv = 0;

	parameter picture_size             = 0;
	parameter convolution_size_1by1    = 0;
	parameter convolution_size_3by3 	= 0;
	
	parameter SIZE_1 = 0;
	parameter SIZE_2 = 0;
	parameter SIZE_3 = 0;
	parameter SIZE_4 = 0;
	parameter SIZE_5 = 0;
	parameter SIZE_6 = 0;
	parameter SIZE_7 = 0;
	parameter SIZE_8 = 0;
	parameter SIZE_9 = 0;
	parameter SIZE_10 = 0;
	parameter SIZE_11 = 0;
	parameter SIZE_12 = 0;
	
	parameter SIZE_address_pix = 0;
	parameter SIZE_address_wei = 0;

	inout 										clk;		       // clock
	input 		signed 	[SIZE_1-1:0] 			data_p, data_w;    // read pixel/weight data in database
	output				[15:0] 					address;	       // pixel/weight address in databse  
	output 	reg 								we_p; 		       // write pixel signal for RAM
	output 	reg 								we_w;		       // write weight signal for RAM
	inout 										re_RAM_w;	       // read RAM signal, read weight to RAM from database
	inout 										re_RAM_p;	       // read RAM signal, read pixel to RAM from database
	input 										nextstep;	       // step + 1
	output	reg	signed	[SIZE_1-1:0] 			dp;   		       // write pixel to RAM from database 
	output 	reg signed 	[SIZE_12-1:0] 			dw;     	       // write weight to RAM from database, weight 3x3
	output	reg			[SIZE_address_pix-1:0]	addrp;		       // pixel data to RAM from database
	output 	reg 		[SIZE_address_wei-1:0] 	addrw;		       // weight to RAM from database
	output 				[6:0] 					step_out;	       // layer step
	input 										GO;			       // start CNN
	input          		[6:0] 					i_f;      	       // input feature map count - 1		  
			  
	reg    [SIZE_address_pix-1:0]  addr;
	wire   [14:0]                  firstaddr,lastaddr;
	reg                            sh;
	reg    [6:0]                   step;
	reg    [6:0]                   step_n;
	reg    [5:0]                   weight_case;            // kernel size
	reg    [SIZE_12-1:0]           buff;		           // weight buffer
	reg    [14:0]                  i;				       // pixel address index
	reg    [14:0]                  i1;				       // weight address index
	
	addressRAM #(.picture_size(picture_size), .convolution_size_1by1(convolution_size_1by1), .convolution_size_3by3(convolution_size_3by3)) addressRAM(
	   .step(step_out), .re_RAM_w(re_RAM_w), .re_RAM_p(re_RAM_p), .firstaddr(firstaddr), .lastaddr(lastaddr));
	
	initial sh			= 0;	// help if statement
	initial weight_case	= 0;	// kernel size
	initial i			= 0;	// pixel address index
	initial i1			= 0;	// weight address index
	
	always @(posedge clk) begin
		if(GO==1) begin 	// step = 1, start CNN
			step=1;	
		end
		  
		sh=sh+1;
			
		if((step_out==1)||(step_out==2)) begin	           // picture
			if((i<=lastaddr-firstaddr)&&(sh==0)) begin		
				addr = i;
				if ((step_out==1)||(step_out==2)) begin	    // write pixel signal for RAM => write pixel in RAM
					we_p = 1;
				end
			end
			if((i<=lastaddr-firstaddr)&&(sh==1)) begin		// write pixel to RAM from database
				if(we_p) begin
					addrp			= firstaddr + addr;
					dp				= 0;
					dp[SIZE_1-1:0]	= data_p;
					we_p 			= 0;
				end
				i = i + 1;					                // address + 1
			end
			 if((i>lastaddr-firstaddr)&&(sh==1))begin
				step = step + 1;                            //next step
				i    = 0;
			end
		end
		
		if((step_out==3)||(step_out==7)||(step_out==11)||(step_out==15)||(step_out==19)||(step_out==23)||(step_out==27)||(step_out==31)||(step_out==37)||(step_out==41)||(step_out==45)||(step_out==49)) begin            // DW weight
			if((i<=lastaddr-firstaddr)&&(sh==0)) begin		
				addr = i1;
			end
			if((i<=lastaddr-firstaddr)&&(sh==1)) begin
				we_w  = 0;
				addrw = addr;								// weight address
				if(weight_case!=0) begin		
					i = i+1;								// kernel block truncation
				end
				
				case (weight_case)							// 3X3 kernel => 9
					0: ;
					1: begin 
						buff=0; 
						buff[SIZE_9-1:SIZE_8] = data_w; 
					end   
					2: buff[SIZE_8-1:SIZE_7] = data_w; 
					3: buff[SIZE_7-1:SIZE_6] = data_w;  
					4: buff[SIZE_6-1:SIZE_5] = data_w;  
					5: buff[SIZE_5-1:SIZE_4] = data_w;           
					6: buff[SIZE_4-1:SIZE_3] = data_w;  
					7: buff[SIZE_3-1:SIZE_2] = data_w;  
					8: buff[SIZE_2-1:SIZE_1] = data_w;   
					9: begin 
						buff[SIZE_1-1:0] = data_w;  
						i1 = i1 + 1; 
					end
					default: $display("Check weight_case");
				endcase
				if (weight_case==9) begin 				  // write 3X3 weight in RAM
					weight_case	= 1; 
					dw			= buff; 
					we_w		= 1; 
				end else begin
					weight_case = weight_case+1;
				end			
			end
			if ((i>lastaddr-firstaddr)&&(sh==1)) begin   // next step
				step		= step + 1;         
				i			= 0;				// kernel block truncation
				i1			= 0;				// weight address index
				weight_case = 0;				// kernel size
			end
		end else if((step_out==5)||(step_out==9)||(step_out==13)||(step_out==17)||(step_out==21)||(step_out==25)||(step_out==29)||(step_out==33)||(step_out==35)||(step_out==39)||(step_out==43)||(step_out==47)||(step_out==51)) begin       // pw weight
			if((i<=lastaddr-firstaddr)&&(sh==0)) begin		
				addr = i1;
			end
			if((i<=lastaddr-firstaddr)&&(sh==1)) begin
				we_w  = 0;
				addrw = addr;								   // weight address
				if(weight_case!=0) begin		
					i = i+1;								   // kernel block truncation
				end
				
				if(i_f==0) begin
                    case(weight_case)							// 1X1 kernel => 1
                        0: ;
                        1: begin 
                            buff=0; 
                            buff[SIZE_1-1:0] = data_w;
                            i1 = i1 + 1;  
                        end 			
                        default: $display("Check weight_case");
				    endcase
                    if(weight_case==1) begin 				   // write 1X1 weight in RAM
                        weight_case	= 1; 
                        dw			= buff; 
                        we_w		= 1; 
                    end else begin
                        weight_case = weight_case+1;
                    end		
				end
				
				if(i_f==2) begin
                    case(weight_case)							// 1X1 kernel => 1
                        0: ;
                        1: begin 
                            buff=0; 
                            buff[SIZE_1-1:0] = data_w;  
                        end
                        2: buff[SIZE_2-1:SIZE_1] = data_w; 
	           			3: begin
	           			     buff[SIZE_3-1:SIZE_2] = data_w;   			
                             i1 = i1 + 1;
                        end
                        default: $display("Check weight_case");
				    endcase
                    if(weight_case==3) begin 				// write 1X1 weight in RAM
                        weight_case	= 1; 
                        dw			= buff; 
                        we_w		= 1; 
                    end else begin
                        weight_case = weight_case+1;
                    end		
				end
				
				if(i_f==5) begin
                    case(weight_case)							// 1X1 kernel => 1
                        0: ;
                        1: begin 
                            buff=0; 
                            buff[SIZE_1-1:0] = data_w;  
                        end
                        2: buff[SIZE_2-1:SIZE_1] = data_w; 
	           			3: buff[SIZE_3-1:SIZE_2] = data_w;   			
                        4: buff[SIZE_4-1:SIZE_3] = data_w;  
					    5: buff[SIZE_5-1:SIZE_4] = data_w;  
                        6: begin
                            buff[SIZE_6-1:SIZE_5] = data_w;
                            i1 = i1 + 1;
                        end
                        default: $display("Check weight_case");
				    endcase
                    if(weight_case==6) begin 				// write 1X1 weight in RAM
                        weight_case	= 1; 
                        dw			= buff; 
                        we_w		= 1; 
                    end else begin
                        weight_case = weight_case+1;
                    end		
				end
				
				if(i_f==8) begin
                    case(weight_case)							// 1X1 kernel => 1
                        0: ;
                        1: begin 
                            buff=0; 
                            buff[SIZE_1-1:0] = data_w;  
                        end
                        2: buff[SIZE_2-1:SIZE_1] = data_w; 
	           			3: buff[SIZE_3-1:SIZE_2] = data_w;   			
                        4: buff[SIZE_4-1:SIZE_3] = data_w;  
					    5: buff[SIZE_5-1:SIZE_4] = data_w;  
                        6: buff[SIZE_6-1:SIZE_5] = data_w;  
					    7: buff[SIZE_7-1:SIZE_6] = data_w;  
					    8: buff[SIZE_8-1:SIZE_7] = data_w;
                        9: begin
                            buff[SIZE_9-1:SIZE_8] = data_w;
                            i1 = i1 + 1;
                        end
                        default: $display("Check weight_case");
				    endcase
                    if(weight_case==9) begin 				// write 1X1 weight in RAM
                        weight_case	= 1; 
                        dw			= buff; 
                        we_w		= 1; 
                    end else begin
                        weight_case = weight_case+1;
                    end		
				end
				
				if(i_f==11) begin
                    case(weight_case)							// 1X1 kernel => 1
					0: ;
					1: begin 
						buff=0; 
						buff[SIZE_1-1:0] = data_w; 
					end   
					2: buff[SIZE_2-1:SIZE_1] = data_w; 
					3: buff[SIZE_3-1:SIZE_2] = data_w;  
					4: buff[SIZE_4-1:SIZE_3] = data_w;  
					5: buff[SIZE_5-1:SIZE_4] = data_w;           
					6: buff[SIZE_6-1:SIZE_5] = data_w;  
					7: buff[SIZE_7-1:SIZE_6] = data_w;  
					8: buff[SIZE_8-1:SIZE_7] = data_w;   
					9: buff[SIZE_9-1:SIZE_8] = data_w;   
					10: buff[SIZE_10-1:SIZE_9] = data_w;   
					11: buff[SIZE_11-1:SIZE_10] = data_w;      
					12: begin 
						buff[SIZE_12-1:SIZE_11] = data_w;  
						i1 = i1 + 1; 
					end
                        default: $display("Check weight_case");
				    endcase
                    if(weight_case==12) begin 				// write 1X1 weight in RAM
                        weight_case	= 1; 
                        dw			= buff; 
                        we_w		= 1; 
                    end else begin
                        weight_case = weight_case+1;
                    end		
				end
			end
			if ((i>lastaddr-firstaddr)&&(sh==1)) begin // next step
				step		= step + 1;         
				i			= 0;				// kernel block truncation
				i1			= 0;				// weight address index
				weight_case = 0;				// kernel size
				if(weight_case < 12) begin
				    weight_case	= 1; 
					dw			= buff; 
					we_w		= 1; 
				end
			end
		end else begin
			we_w=0;
		end
	end
	
	always @(posedge nextstep) begin
		if (GO==1) begin
			step_n = 0; 
		end	else begin
			step_n = step_n+1;
		end
	end
	
	assign step_out = step + step_n;
	assign address  = firstaddr + i;
endmodule

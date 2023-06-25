module addressRAM(
	input 		[6:0] 	step,				                            // conv layer step
	output	reg			re_RAM_p, re_RAM_w,               				// read RAM signal, read pixel/weight from database
	output	reg	[14:0]	firstaddr, lastaddr                          	// start/end address
	);	
	
	parameter picture_size		       = 0;
	parameter convolution_size_1by1   = 0;
	parameter convolution_size_3by3   = 0;


	parameter picture_storage_limit_image1 = picture_size*picture_size;                                // image1 0~1599  
	parameter picture_storage_limit_image2 = picture_storage_limit_image1+ picture_size*picture_size;  // image2 1600 ~ 3199
    
	
	parameter conv1 =  1 * convolution_size_3by3;                          // DW1, image1  0 ~ 8
	parameter conv2 = conv1 + 3 * convolution_size_1by1;                   // PW1, image1  9 ~ 11
	parameter conv3 = conv2 + 3 * convolution_size_3by3;                   // DW2, image1  12 ~ 38
	parameter conv4 = conv3 + 9 * convolution_size_1by1;                   // PW2, image1  39 ~ 47
	parameter conv5 = conv4 + 6 * convolution_size_3by3;                   // DW3, image1  48 ~ 101
	parameter conv6 = conv5 + 18 * convolution_size_1by1;                  // PW3, image1  102 ~ 119
	parameter conv7 = conv6 + 9 * convolution_size_3by3;                   // DW4, image1  120 ~ 200
	parameter conv8 = conv7 + 27 * convolution_size_1by1;                  // PW4, image1  201 ~ 227
    
    
    parameter conv9 = conv8 + 1 * convolution_size_3by3;                   // DW1, image2  228 ~ 236
	parameter conv10 = conv9 + 3 * convolution_size_1by1;                  // PW1, image2  237 ~ 239
	parameter conv11 = conv10 + 3 * convolution_size_3by3;                 // DW2, image2  240 ~ 266
	parameter conv12 = conv11 + 9 * convolution_size_1by1;                 // PW2, image2  267 ~ 275
	parameter conv13 = conv12 + 6 * convolution_size_3by3;                 // DW3, image2  276 ~ 329
	parameter conv14 = conv13 + 18 * convolution_size_1by1;                // PW3, image2  330 ~ 347
	parameter conv15 = conv14 + 9 * convolution_size_3by3;                 // DW4, image2  348 ~ 428
	parameter conv16 = conv15 + 27 * convolution_size_1by1;                // PW4, image2  429 ~ 455
	
	parameter conv17 = conv16 + 12 * convolution_size_3by3;                // DW5, decoder  456 ~ 563
	parameter conv18 = conv17 + 144 * convolution_size_1by1;               // PW5, decoder  564 ~ 707
	parameter conv19 = conv18 + 12 * convolution_size_3by3;                // DW6, decoder  708 ~ 815
	parameter conv20 = conv19 + 72 * convolution_size_1by1;                // PW6, decoder  816 ~ 887
	parameter conv21 = conv20 + 6 * convolution_size_3by3;                 // DW7, decoder  888 ~ 941
	parameter conv22 = conv21 + 18 * convolution_size_1by1;                // PW7, decoder  942 ~ 959
	parameter conv23 = conv22 + 3 * convolution_size_3by3;                 // DW8, decoder  960 ~ 986
	parameter conv24 = conv23 + 3 * convolution_size_1by1;                 // PW8, decoder  987 ~ 989
	

	always @(step) begin
		case (step)
		1'd1: begin               // picture
				firstaddr 	= 0;
				lastaddr 	= picture_storage_limit_image1;
				re_RAM_p 		= 1;
			  end 
		2'd2: begin               // picture
				firstaddr 	= picture_storage_limit_image1;
				lastaddr 	= picture_storage_limit_image2;
				re_RAM_p 		= 1;
			  end
		2'd3: begin               // weights DW1, image1
				firstaddr 	= 0;
				lastaddr 	= conv1;
				re_RAM_p 		= 0;
				re_RAM_w 		= 1;
			  end
		3'd5: begin               // weights PW1, image1
				firstaddr 	= conv1;
				lastaddr 	= conv2;
				re_RAM_w 		= 1;
			  end		
		3'd7: begin	              // weights DW2, image1
				firstaddr 	= conv2;
				lastaddr 	= conv3;
				re_RAM_w		= 1;
			  end
		4'd9: begin		          // weights PW2, image1
				firstaddr 	= conv3;
				lastaddr 	= conv4;
				re_RAM_w 		= 1;
				end
		4'd11: begin		     // weights DW3, image1
				firstaddr 	= conv4;
				lastaddr 	= conv5;
				re_RAM_w 		= 1;
			  end
		4'd13: begin		     // weights PW3, image1
				firstaddr 	= conv5;
				lastaddr 	= conv6;
				re_RAM_w		= 1;
			  end
		4'd15: begin		     // weights DW4, image1
				firstaddr 	= conv6;
				lastaddr 	= conv7;
				re_RAM_w 		= 1;
			  end
		5'd17: begin		     // weights PW4, image1
				firstaddr 	= conv7;
				lastaddr 	= conv8;
				re_RAM_w 		= 1;
			  end
		5'd19: begin		     // weights DW1, image2
				firstaddr 	= conv8;
				lastaddr 	= conv9;
				re_RAM_w 		= 1;
			  end
		5'd21: begin		     // weights PW1, image2
				firstaddr 	= conv9;
				lastaddr 	= conv10;
				re_RAM_w 		= 1;
			  end
		5'd23: begin		     // weights DW2, image2
				firstaddr 	= conv10;
				lastaddr 	= conv11;
				re_RAM_w 		= 1;
			  end
		5'd25: begin		     // weights PW2, image2
				firstaddr 	= conv11;
				lastaddr 	= conv12;
				re_RAM_w 		= 1;
			  end
		5'd27: begin		     // weights DW3, image2
				firstaddr 	= conv12;
				lastaddr 	= conv13;
				re_RAM_w 		= 1;
			  end    
		5'd29: begin		     // weights PW3, image2
				firstaddr 	= conv13;
				lastaddr 	= conv14;
				re_RAM_w 		= 1;
			  end
		5'd31: begin		     // weights DW4, image2
				firstaddr 	= conv14;
				lastaddr 	= conv15;
				re_RAM_w 		= 1;
			  end
		6'd33: begin		     // weights PW4, image2
				firstaddr 	= conv15;
				lastaddr 	= conv16;
				re_RAM_w 		= 1;
			  end
		6'd35: begin		      // fusion layer
				re_RAM_w 		= 0;
		    end   	  
	    6'd37: begin		      // weights DW1, decoder
				firstaddr 	= conv16;
				lastaddr 	= conv17;
				re_RAM_w 		= 1;
			  end
	    6'd39: begin		       // weights PW1, decoder
				firstaddr 	= conv17;
				lastaddr 	= conv18;
				re_RAM_w 		= 1;
			  end
	    6'd41: begin		       // weights DW2, decoder
				firstaddr 	= conv18;
				lastaddr 	= conv19;
				re_RAM_w 		= 1;
			  end
		6'd43: begin		       // weights PW2, decoder
				firstaddr 	= conv19;
				lastaddr 	= conv20;
				re_RAM_w 		= 1;
			  end
		 6'd45: begin		       // weights DW3, decoder
				firstaddr 	= conv20;
				lastaddr 	= conv21;
				re_RAM_w 		= 1;
			  end
		 6'd47: begin		       // weights PW3, decoder
				firstaddr 	= conv21;
				lastaddr 	= conv22;
				re_RAM_w 		= 1;
			  end
		 6'd49: begin		       // weights DW4, decoder
				firstaddr 	= conv22;
				lastaddr 	= conv23;
				re_RAM_w 		= 1;
			  end
		 6'd51: begin		       // weights PW4, decoder
				firstaddr 	= conv23;
				lastaddr 	= conv24;
				re_RAM_w 		= 1;
			  end  	  	  	  	  	  	  	  	  
		default: begin
					re_RAM_w = 0;
					re_RAM_p = 0;
				end
		endcase
	end
endmodule

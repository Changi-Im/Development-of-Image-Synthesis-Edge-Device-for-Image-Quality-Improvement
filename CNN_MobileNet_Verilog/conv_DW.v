module conv_DW(clk, Y1, prov, matrix, matrix2, i, w1, w2, w3, w4, w5, w6, w7, w8, w9,w11, w12, w13, w14, w15, w16, w17, w18, w19, conv_DW_en);
	
	parameter SIZE = 0;
	
	input 								clk;											// clock
	output	reg	signed	[SIZE+SIZE-2:0]	Y1;												// 3x3 kernel calculatioin result (MAC calculatioin)
	input 				[1:0]			prov;											// 2'b00 : center, 2'b11 : left, 2'b10 : right
	input 				[6:0] 			matrix;      									// feature map size(line) 
	input 				[12:0] 			matrix2;										// matrix * matrix
	input 				[14:0] 			i;												// current index of feature map pixel
	input 		signed 	[SIZE-1:0] 		w1, w2, w3, w4, w5, w6, w7, w8, w9;				// pixel
	input 		signed 	[SIZE-1:0] 		w11, w12, w13, w14, w15, w16, w17, w18, w19;	// weight
	input 								conv_DW_en;										// convolution MAC calculatioin enable signal
    
    //===================================
    //== w9, w19 == w7, w17 == w5, w15 ==
    //===================================
    //== w3, w13 == w1, w11 = w2, w12  ==
    //===================================
    //== w4, w14 == w6, w16 = w8, w18  ==
    //===================================
	
	// we use zero padding
	
	always @(posedge clk) begin
		if(conv_DW_en==1) begin
			Y1 = 0;
			Y1 = Y1+Y(w1,w11);

			//right
			if(prov!=2'b10) begin
				Y1 = Y1+Y(w2,w12);
			end
			
			//left
			if(prov!=2'b11) begin
				Y1 = Y1+Y(w3,w13);
			end
			
			//downleft
			if((i<matrix2-matrix)&&(prov!=2'b11)) begin
				Y1 = Y1+Y(w4,w14);
			end
			
			//upright
			if((i>matrix-1'b1)&&(prov!=2'b10)) begin
				Y1 = Y1+Y(w5,w15);
			end
			
			//down
			if(i<matrix2-matrix) begin
				Y1 = Y1+Y(w6,w16);
			end
			
			//up
			if(i>matrix-1'b1) begin
				Y1 = Y1+Y(w7,w17);
			end
			
			//downright
			if((i<matrix2-matrix)&&(prov!=2'b10)) begin
				Y1 = Y1+Y(w8,w18);
			end
			
			//upleft
			if((i>matrix-1'b1)&&(prov!=2'b11)) begin
				Y1 = Y1+Y(w9,w19);
			end
		end
	end

	function signed [SIZE+SIZE-2:0] Y;
		input signed [SIZE-1:0] a, b;
		begin
			Y = a*b;
		end
	endfunction
endmodule

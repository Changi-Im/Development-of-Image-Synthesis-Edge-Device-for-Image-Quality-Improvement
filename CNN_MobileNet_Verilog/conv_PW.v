module conv_PW(clk, Y1, matrix, matrix2, i, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, filt, conv_PW_en);
	
	parameter SIZE = 0;

	input 								clk;											      // clock
	output	reg	signed	[SIZE+SIZE-2:0]	Y1;												      // Point Wise 1X1 calculatioin result (MAC calculatioin)
	input 				[6:0] 			matrix;      									      // feature map size(line) 
	input 				[12:0] 			matrix2;										      // matrix * matrix
	input 				[14:0] 			i;												      // current index of feature map pixel
	input 		signed 	[SIZE-1:0] 		p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;    // pixel
	input 		signed 	[SIZE-1:0] 		w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12;	  // weight
	input 								conv_PW_en;										      // Point Wise MAC calculatioin enable signal
	input               [6:0]           filt;                                                 // input feature map count - 1
	
	always @(posedge clk) begin
		if(conv_PW_en==1) begin
            Y1 = 0;
            Y1 = Y1 + Y(p1,w1);
            
            if(filt>=1) begin
                Y1 = Y1 + Y(p2, w2);
            end
            
            if(filt>=2) begin
                Y1 = Y1 + Y(p3, w3);
            end
            
            if(filt>=5) begin
                Y1 = Y1 + Y(p4, w4);
                Y1 = Y1 + Y(p5, w5);
                Y1 = Y1 + Y(p6, w6);
            end
            
            if(filt>=8) begin
                Y1 = Y1 + Y(p7, w7);
                Y1 = Y1 + Y(p8, w8);
                Y1 = Y1 + Y(p9, w9);
            end
            
            if(filt>=11) begin
                Y1 = Y1 + Y(p10, w10);
                Y1 = Y1 + Y(p11, w11);
                Y1 = Y1 + Y(p12, w12);
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

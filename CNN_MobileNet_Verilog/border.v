module border(
    input 				clk, go,	// clock, border enabel signal(conv_DW_en)
    input 		[14:0] 	i,			// pixel location
    input 		[6:0] 	matrix,		// feature map size
    output	reg	[1:0]	prov		// 2'b00 : center / 2'b11 : left / 2'b10 : right
	);
	
	reg [10:0] j;
	
	always @(posedge clk) begin
		if(go==1) begin
			prov = 0;
			for(j=1'b1; j<=matrix; j=j+1'b1) begin
				if((i==j*matrix-1'b1)&&(prov!=2'b10)) begin
				    prov = 2'b10;
				end
				if(((i==0)||(i==j*matrix))&&(prov!=2'b11)) begin
					prov = 2'b11;
				end
			end
		end	else begin
			prov = 0;
		end
	end
endmodule
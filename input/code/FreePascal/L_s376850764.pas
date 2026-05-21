var
	n,m,i,a:Longint;
	fiv,six:Char;
	ffiv,fsix:Boolean;
	used:array[1..9]of Boolean;
	S:String;
begin
	read(n,m);
	for i:=1 to m do begin
		read(a);
		used[a]:=true;
	end;
	if used[5]then begin
		fiv:='5';
		ffiv:=true;
	end else if used[3]then begin
		fiv:='3';
		ffiv:=true;
	end else if used[2]then begin
		fiv:='2';
		ffiv:=true;
	end;
	if used[9]then begin
		six:='9';
		fsix:=true;
	end else if used[6]then begin
		six:='6';
		fsix:=true;
	end;
	if used[1]then begin
		if n mod 2=1 then begin
			if used[7]then begin
				dec(n,3);
				write('7');
			end else if ffiv then begin
				dec(n,5);
				write(fiv);
			end else if used[8]then begin
				dec(n,7);
				write('8');
			end;
		end;
		for i:=1 to n div 2 do write('1');
	end else if used[7]then begin
		if n mod 3=2 then begin
			if ffiv then begin
				dec(n,5);
				S:=fiv;
			end else if used[4]then begin
				dec(n,8);
				S:='44';
			end else if used[8]then begin
				dec(n,14);
				write('88');
			end;
		end else if n mod 3=1 then begin
			if used[4]then begin
				dec(n,4);
				S:='4';
			end else if used[8]then begin
				dec(n,7);
				write('8');
			end else if ffiv then begin
				dec(n,10);
				S:=fiv+fiv;
			end;
		end;
		for i:=1 to n div 3 do write('7');
		write(S);
	end else if used[4]then begin
		if n mod 4=3 then begin
			if used[9]and ffiv and(n>=11)then begin
				dec(n,11);
				write('9');
				if used[5]then begin
					write('5');
				end else begin
					S:=fiv;
				end;
			end else if used[8]then begin
				dec(n,7);
				write('8');
			end else if fsix and ffiv then begin
				dec(n,11);
				write(six);
				if used[5]then begin
					write('5');
				end else begin
					S:=fiv;
				end;
			end else if ffiv then begin
				dec(n,15);
				if used[5]then begin
					write('555');
				end else begin
					S:=fiv+fiv+fiv;
				end;
			end;
		end else if n mod 4=2 then begin
			if fsix then begin
				dec(n,6);
				write(six);
			end else if ffiv then begin
				dec(n,10);
				if used[5]then begin
					write('55');
				end else begin
					S:=fiv+fiv;
				end;
			end else if used[8]then begin
				dec(n,14);
				write('88');
			end;
		end else if n mod 4=1 then begin
			if ffiv then begin
				dec(n,5);
				if used[5]then begin
					write('5');
				end else begin
					S:=fiv;
				end;
			end else if used[8]then begin
				if used[9]then begin
					dec(n,13);
					write('98');
				end else if used[6]then begin
					dec(n,13);
					write('86');
				end else begin
					dec(n,21);
					write('888');
				end;
			end;
		end;
		for i:=1 to n div 4 do write('4');
		write(S);
	end else if ffiv then begin
		if n mod 5=4 then begin
			if used[9] and(n>=24)then begin
				dec(n,24);
				write('9999');
			end else if used[8]then begin
				dec(n,14);
				write('88');
			end else if used[6]then begin
				dec(n,24);
				write('6666');
			end;
		end else if n mod 5=3 then begin
			if used[9]and(n>=18)then begin
				dec(n,18);
				write('999');
			end else if fsix and used[8]then begin
				dec(n,13);
				if used[9]then begin
					write('98');
				end else begin
					write('86');
				end;
			end else if used[6]then begin
				dec(n,18);
				write('666');
			end else if used[8]then begin
				dec(n,28);
				write('8888');
			end;
		end else if n mod 5=2 then begin
			if used[9]and(n>=12)then begin
				dec(n,12);
				write('99');
			end else if used[8]then begin
				dec(n,7);
				write('8');
			end else if used[6]then begin
				dec(n,12);
				write('66');
			end;
		end else if n mod 5=1 then begin
			if fsix then begin
				dec(n,6);
				write(six);
			end else if used[8]then begin
				dec(n,21);
				write('888');
			end;
		end;
		for i:=1 to n div 5 do write(fiv);
	end else if fsix then begin
		a:=n mod 6;
		if used[9]then begin
			for i:=1 to (n-a*7)div 6 do write('9');
			for i:=1 to a do write('8');
		end else begin
			for i:=1 to a do write('8');
			for i:=1 to (n-a*7)div 6 do write('6');
		end;
	end else if used[8]then begin
		for i:=1 to n div 7 do write('8');
	end;
end.
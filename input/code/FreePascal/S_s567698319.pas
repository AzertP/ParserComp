var
	L,R,ans,tmp,M,i,j:Longint;
begin
	read(L,R);
	M:=2019;
	ans:=M;
	if L+2018<=R then begin
		writeln(0);
		exit;
	end;
	for i:=L to R do begin
		for j:=1 to 100 do begin
			if i+j>R then break;
			tmp:=i mod M*(i+j)mod M;
			if ans>tmp then ans:=tmp;
		end;
	end;
	writeln(ans);
end.

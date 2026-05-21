var
	N,K,i,u,v:Longint;
	S:String;
procedure f(u:Longint);
begin
	if u=100 then S:=S+'100'
	else begin
		if u>9 then S:=S+chr(u div 10+48);
		S:=S+chr(u mod 10+48);
	end;
end;
begin
	read(N,K);
	if (N-1)*(N-2)<K*2 then begin
		writeln(-1);
		exit;
	end;
	K:=(N-1)*(N-2)div 2-K;
	writeln(K+N-1);
	for i:=2 to N do begin
		S:=S+'1 ';
		f(i);
		S:=S+' ';
	end;
	for u:=2 to N do begin
		if K=0 then break;
		for v:=u+1 to N do begin
			if K=0 then break;
			f(u);
			S:=S+' ';
			f(v);
			S:=S+' ';
			dec(K);
		end;
	end;
	writeln(S);
end.

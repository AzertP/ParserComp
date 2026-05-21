var
	N,i,s,a,b:Longint;
	X:Array[1..100]of Longint;
begin
	read(N);
	for i:=1 to N do begin
		read(X[i]);
		inc(s,X[i]);
	end;
	s:=s div N;
	for i:=1 to N do begin
		inc(a,(s-X[i])*(s-X[i]));
		inc(b,(s+1-X[i])*(s+1-X[i]));
	end;
	if a>b then a:=b;
	writeln(a);
end.

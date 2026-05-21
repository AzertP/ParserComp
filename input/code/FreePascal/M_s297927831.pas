var
	N,X,Y,Z,i,p:Longint;
	S:String;
begin
	readln(N,X,X);
	readln(S);
	i:=length(S);
	Z:=0;
	p:=1;
	while(1<=i)and(S[i]<>' ')do begin
		inc(Z,(ord(S[i])-48)*p);
		p:=p*10;
		dec(i);
	end;
	dec(i);
	Y:=0;
	p:=1;
	if N=1 then Y:=X
	else begin
		while(1<=i)and(S[i]<>' ')do begin
			inc(Y,(ord(S[i])-48)*p);
			p:=p*10;
			dec(i);
		end;
	end;
	dec(X,Z);
	dec(Y,Z);
	if X<0 then X:=-X;
	if Y<0 then Y:=-Y;
	if X<Y then X:=Y;
	writeln(X);
end.

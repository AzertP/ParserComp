var
	N,i:Longint;
	A:Array[1..100000]of int64;
	ret:int64;
begin
	read(N);
	for i:=1 to N do begin
		read(A[i]);
		if A[i]=0 then begin
			writeln(0);
			exit;
		end;
	end;
	ret:=1;
	for i:=1 to N do begin
		if A[i]>1000000000000000000 div ret then begin
			writeln(-1);
			exit;
		end;
		ret:=ret*A[i];
	end;
	writeln(ret);
end.

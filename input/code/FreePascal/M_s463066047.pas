var
	N,i,sz:Longint;
	ans,T,K:int64;
	S:String;
	A:Array[1..100]of Char;
	L:Array[1..100]of Longint;
begin
	readln(S);
	readln(K);
	N:=length(S);
	A[1]:=S[1];
	L[1]:=1;
	sz:=1;
	for i:=2 to N do begin
		if A[sz]<>S[i]then begin
			inc(sz);
			A[sz]:=S[i];
			L[sz]:=1;
		end else inc(L[sz]);
	end;
	if sz=1 then begin
		writeln(K*L[1]div 2);
		exit;
	end;
	inc(ans,L[1]div 2);
	inc(ans,L[sz]div 2);
	T:=0;
	for i:=2 to sz-1 do inc(T,L[i]div 2);
	inc(ans,T*K);
	if A[1]=A[sz]then T:=(L[1]+L[sz])div 2
	else T:=L[1]div 2+L[sz]div 2;
	inc(ans,T*(K-1));
	writeln(ans);
end.

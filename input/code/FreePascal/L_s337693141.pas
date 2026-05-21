var
	N,i,j,sz:Longint;
	A:Array[1..100,1..99]of Longint;
	ans:String;
begin
	read(N);
	if N=2 then begin
		writeln(-1);
		exit;
	end;
	if N mod 2=1 then begin
		for i:=1 to N-1 do A[N,i]:=i;
		i:=1;
		while i<=N-1 do begin
			for j:=1 to i-1 do begin
				A[i,j]:=j;
				A[i+1,j]:=j;
			end;
			A[i,i]:=i+1;
			A[i+1,i+1]:=i;
			A[i,i+1]:=i+2;
			A[i+1,i]:=i+2;
			for j:=i+2 to N-1 do begin
				A[i,j]:=j+1;
				A[i+1,j]:=j+1;
			end;
			inc(i,2);
		end;
	end else begin
		i:=1;
		while i<=N-2 do begin
			for j:=1 to i-1 do begin
				A[i,j]:=j;
				A[i+1,j]:=j;
			end;
			A[i,i]:=i+1;
			A[i+1,i+1]:=i;
			A[i,i+1]:=i+2;
			A[i+1,i]:=i+2;
			for j:=i+2 to N-1 do begin
				A[i,j]:=j+1;
				A[i+1,j]:=j+1;
			end;
			inc(i,2);
		end;
		for j:=1 to N-3 do begin
			A[N-1,j]:=j;
			A[N,j]:=j;
		end;
		A[N-1,N-2]:=N;
		A[N,N-1]:=N-1;
		A[N-1,N-1]:=N-2;
		A[N,N-2]:=N-2;
	end;
	setlength(ans,3*N*N);
	sz:=0;
	for i:=1 to N do begin
		for j:=1 to N-1 do begin
			if A[i,j]<10 then begin
				inc(sz);
				ans[sz]:=Chr(A[i,j]+48);
			end else if A[i,j]<100 then begin
				inc(sz);
				ans[sz]:=Chr(A[i,j]div 10+48);
				inc(sz);
				ans[sz]:=Chr(A[i,j]mod 10+48);
			end else begin
				inc(sz);
				ans[sz]:='1';
				inc(sz);
				ans[sz]:='0';
				inc(sz);
				ans[sz]:='0';
			end;
			inc(sz);
			ans[sz]:=#10;
		end;
	end;
	setlength(ans,sz);
	write(ans);
end.

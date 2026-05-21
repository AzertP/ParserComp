var
	H,W,i,j,x:Longint;
	S,T:String[100];
	dp:Array[1..100]of Longint;
begin
	readln(H,W);
	S[1]:='.';
	for j:=2 to W do dp[j]:=100000;
	for i:=1 to H do begin
		readln(T);
		for j:=1 to W do if S[j]>T[j]then inc(dp[j]);
		S:=T;
		for j:=2 to W do begin
			x:=dp[j-1];
			if S[j-1]>S[j]then inc(x);
			if x<dp[j]then dp[j]:=x;
		end;
	end;
	writeln(dp[W]);
end.
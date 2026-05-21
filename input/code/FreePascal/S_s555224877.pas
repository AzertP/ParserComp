var
	N,i,a,j,ni,nj:Longint;
	dp:Array[0..200000,0..3]of int64;
begin
read(N);
for i:=0 to N do for j:=0 to 2 do dp[i,j]:=-1000000000000000000;
dp[0,0]:=0;
for i:=0 to N-1 do begin
read(a);
for j:=0 to 2 do begin
if dp[i+1,j+1]<dp[i,j]then dp[i+1,j+1]:=dp[i,j];
ni:=i+2;if ni>N then ni:=N;
nj:=j;if i+2=N then inc(nj);
if dp[ni,nj]<dp[i,j]+a then dp[ni,nj]:=dp[i,j]+a;
end;
end;
writeln(dp[N,1+N mod 2]);
end.

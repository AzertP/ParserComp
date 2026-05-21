var N,i,P,ans:Longint;A:Array[1..100]of Longint;
begin
read(N);
P:=1;
for i:=1 to N do begin
read(A[i]);
if A[P]<>A[i]then begin
inc(ans,(i-P)div 2);
P:=i;
end;
end;
writeln(ans+(N+1-P)div 2);
end.
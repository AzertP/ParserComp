var n,i,m,s,x:Longint;
begin
read(n);
m:=0;
for i:=1 to n do begin
read(x);
inc(s,x);
if m<x then m:=x;
end;
writeln(s-m div 2);
end.
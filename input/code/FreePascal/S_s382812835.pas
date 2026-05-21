var
n,i,j,cnt,t:Longint;
a,b:array[1..10000]of Longint;
begin
read(n);
t:=n+1-n mod 2;
for i:=1 to n do begin
for j:=i+1 to t-i-1 do begin
inc(cnt);
a[cnt]:=i;
b[cnt]:=j;
end;
if(j=t-i-1)or(j=t-i)then j:=t-i+1;
for j:=j to n do begin
inc(cnt);
a[cnt]:=i;
b[cnt]:=j;
end;
end;
writeln(cnt);
for i:=1 to cnt do writeln(a[i],' ',b[i]);
end.
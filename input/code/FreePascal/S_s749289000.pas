var
a:array[0..55] of integer;
n,m,i,j,t:integer;
s:longint;
begin
readln(n,m);
for i:=1 to n do read(a[i]);
for i:=1 to n-1 do
  for j:=i+1 to n do
    if a[i]<a[j] then
      begin
        t:=a[i];
        a[i]:=a[j];
        a[j]:=t;
      end;
for i:=1 to m do s:=s+a[i];
write(s);
end.
program canh_toan;
uses math;
const minn=0;
      maxn=trunc(1e5)+5;
var n:longint;
    k:int64;
    a,b:array [minn..maxn] of int64;
procedure sort(l,r:int64);
var i,j,k,tg:int64;
begin
i:=l;j:=r;tg:=a[random(r-l+1)+l];
repeat
while (a[i]<tg) do inc(i);
while (tg<a[j]) do dec(j);
if not(i>j) then
   begin
     k:=a[i];a[i]:=a[j];a[j]:=k;
     k:=b[i];b[i]:=b[j];b[j]:=k;
     inc(i);
     dec(j);
   end;
until i>j;
if (l<j) then sort(l,j);
if (i<r) then sort(i,r);
end;
procedure main();
var i:longint;
begin
readln(n,k);
for i:=1 to n do readln(a[i],b[i]);
sort(1,n);
b[0]:=0;
for i:=1 to n do b[i]:=b[i]+b[i-1];
i:=1;
while (b[i]<k) do inc(i);
write(a[i]);
end;
BEGIN
  main();
END.
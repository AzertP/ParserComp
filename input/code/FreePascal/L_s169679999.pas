program canh_toan;
uses math;
const minn=0;
      maxn=trunc(1e5)+5;
      vao='CONNECT.INP';
      ra='CONNECT.OUT';
      module=trunc(1e9)+7;
var n:longint;
    a,b:array[minn..maxn] of longint;
procedure sorta(l,r:longint);
var i,j,k,tg:longint;
begin
i:=l;j:=r;tg:=a[random(r-l+1)+l];
repeat
while (a[i]<tg) do inc(i);
while (tg<a[j]) do dec(j);
if not(i>j) then
   begin
     k:=a[i];
     a[i]:=a[j];
     a[j]:=k;
     inc(i);
     dec(j);
   end;
until i>j;
if (l<j) then sorta(l,j);
if (i<r) then sorta(i,r);
end;
procedure sortb(l,r:longint);
var i,j,k,tg:longint;
begin
i:=l;j:=r;tg:=b[random(r-l+1)+l];
repeat
while (b[i]<tg) do inc(i);
while (tg<b[j]) do dec(j);
if not(i>j) then
   begin
     k:=b[i];
     b[i]:=b[j];
     b[j]:=k;
     inc(i);
     dec(j);
   end;
until i>j;
if (l<j) then sortb(l,j);
if (i<r) then sortb(i,r);
end;
function nhannp(x,y:int64):int64;
var tg:int64;
begin
if y=0 then exit(0)
else if y=1 then exit(x mod module)
else
  begin
    tg:=nhannp(x,y div 2) mod module;
    tg:=(2*tg) mod module;
    if y mod 2=1 then tg:=(tg+x) mod module;
    exit(tg mod module);
  end;
end;
procedure main();
var i,j,x,y:longint;
    kq:int64;
begin
readln(n);
for i:=1 to n do read(a[i]);
readln;
for j:=1 to n do read(b[j]);
sorta(1,n);sortb(1,n);
i:=1;j:=1;x:=0;y:=0;kq:=1;
while (i<=n) and (j<=n) do
  begin
    if (a[i]<b[j]) then
       begin
         y:=x+1;
         inc(i);
       end
    else
       begin
         y:=x-1;
         inc(j);
       end;
    if (abs(x)>abs(y)) then kq:=nhannp(kq,abs(x)) mod module;
    x:=y;
  end;
while (i<=n) do
  begin
    kq:=nhannp(kq,abs(x)) mod module;
    inc(x);
    inc(i);
  end;
while (j<=n) do
  begin
    kq:=nhannp(kq,abs(x)) mod module;
    dec(x);
    inc(j);
  end;
write(kq);
end;
BEGIN
  main();
END.
var n,h,num,max:int64;
    i:longint;
    a,b:array[0..200000]of int64;
procedure qs(l,r:longint);
var i,j,m,t:int64;
begin
  i:=l;
  j:=r;
  m:=b[(l+r)>>1];
  repeat
    while b[i]>m do inc(i);
    while b[j]<m do dec(j);
    if i<=j then
    begin
      t:=b[i];b[i]:=b[j];b[j]:=t;
      inc(i);
      dec(j);
    end;
  until i>j;
  if l<j then qs(l,j);
  if i<r then qs(i,r);
end;

begin
  read(n,h);
  for i:=1 to n do
  read(a[i],b[i]);
  qs(1,n); 
  for i:=1 to n do
  if a[i]>max then max:=a[i];
  for i:=1 to n do
  begin
    if (b[i]>max) then
    begin
      dec(h,b[i]);
      inc(num);
    end;
    if h<=0 then break;
  end;
  if h>0 then
  begin
    inc(num,h div max);
    if h mod max<>0 then inc(num);
  end;
  writeln(num);
end.

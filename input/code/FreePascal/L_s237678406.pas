var n:longint;
    i,k:longint;
    a,b,c,d,heap:array[0..201000]of int64;
    m0,m1:int64;
    ans,tot,x,y:int64;
procedure swap(var a,b:int64);
var t:longint;
begin
  t:=a;a:=b;b:=t;
end;
procedure up(x:longint);
begin
  while x>1 do
  begin
    if heap[x]<heap[x div 2] then
    begin
      swap(heap[x],heap[x div 2]);
      x:=x div 2;
    end else break;
  end;
end;
procedure down(x:longint);
var min:longint;
begin
  while x*2<=tot do
  begin
    min:=x*2;
    if (x*2+1<=tot)and(heap[x*2+1]<heap[min]) then inc(min);
    if heap[x]>heap[min] then
    begin
      swap(heap[x],heap[min]);
      x:=min;
    end else break;
  end;
end;

procedure sort;
var i,j:longint;
begin
  for i:=1 to n do
  begin
    c[i]:=heap[1];
    heap[1]:=heap[tot];
    dec(tot);
    down(1);
  end;
  {i:=l;
  j:=r;
  m:=c[(l+r)>>1];
  repeat
    while c[i]<m do inc(i);
    while c[j]>m do dec(j);
    if i<=j then
    begin
      t:=c[i];c[i]:=c[j];c[j]:=t;
      inc(i);
      dec(j);
    end;
  until i>j;
  if l<j then qs(l,j);
  if i<r then qs(i,r);}
end;
function find(x:int64):longint;
var l,r,m:longint;
begin
  l:=1;
  r:=n;
  while l<=r do
  begin
    m:=(l+r)>>1;
    if c[m]<x then l:=m+1 else r:=m-1;
  end;
  exit(l);
end;

begin
  read(n);
  for i:=1 to n do
  read(a[i]);
  for i:=1 to n do
  read(b[i]);
  c[0]:=-1;
  c[n+1]:=1 << 45;
  for k:=1 to 29 do
  begin
    tot:=0;
    x:=1 << k;
    y:=1 << (k-1);
    for i:=1 to n do
    begin
      heap[i]:=a[i] and (x-1);
      inc(tot);
      up(i);
      d[i]:=b[i] and (x-1);
    end;
    //writeln(k);
    sort;
    m0:=0;
    m1:=0;
    for i:=1 to n do
    begin
      m1:=m1+find(x-d[i])-find(y-d[i]);
      m1:=m1+n+1-find(x+y-d[i]);
      //inc(m1);
    end;
    //writeln('0=',m0,' 1=',m1);
    if m1 and 1 =1 then ans:=ans+y;
  end;
  writeln(ans);
end.





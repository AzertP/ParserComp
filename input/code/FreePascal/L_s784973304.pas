procedure quickly_sort(var a:array of longint;l,r:longint);inline;
var i,j,x,y:longint;
begin
  while true do
  begin
    if r<l+74 then
      exit;
    i:=l;j:=r;
    x:=a[(l+r)>>1];
    repeat
      while a[i]<x do inc(i);
      while x<a[j] do dec(j);
      if not(i>j) then
      begin
        y:=a[i];a[i]:=a[j];
        a[j]:=y;inc(i);dec(j);
      end;
    until i>j;
    if l<j then quickly_sort(a,l,j);
    if not(i<r) then
      exit;
    l:=i;
  end;
end;
procedure sort(var a:array of longint;l,r:longint);inline;
var i,j,t:longint;
begin
  if r>l then
  begin
    quickly_sort(a,l,r);
    i:=l;
    for i:=l to r do
    begin
      j:=i; 
      while j>pred(l) do
        if a[j]<a[pred(j)] then
        begin
          t:=a[pred(j)];a[pred(j)]:=a[j];a[j]:=t;
          dec(j);
        end
        else
          break;
    end;
  end;
end;
procedure swap(var a,b:longint);inline;
var t:longint;
begin
  t:=a;a:=b;b:=t;
end;
function max(a,b:longint):longint;inline;
begin
  if a>b then
    exit(a);
  exit(b);
end;
function min(a,b:longint):longint;inline;
begin
  if a>b then
    exit(b);
  exit(a);
end;
function endl:ansistring;inline;
begin
  writeln;exit('');
end;
var a:array[0..1000000]of longint;
    n,ans,i:longint;
begin
  readln(n);
  for i:=1 to n do
    read(a[i]);
  sort(a,1,n);
  i:=1;
  while i<n do
    if a[i]=a[i+1] then
    begin
      inc(ans,2);
      inc(i,2);
    end
    else
      inc(i);
  writeln(n-ans);
end.
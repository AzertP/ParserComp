var
  a:array[0..100010]of int64;
  k,l,r,n,ans:int64;
  i:longint;

function min(x,y:int64):int64;
begin
  if x<y then exit(x);
  exit(y);
end;

function max(x,y:int64):int64;
begin
  if x>y then exit(x);
  exit(y);
end;

begin
  readln(n,k);
  for i:=1 to n do
    read(a[i]);
  ans:=maxlongint;
  for i:=1 to n-k+1 do
  begin
    l:=min(a[i],0);
    r:=max(a[i+k-1],0);
    ans:=min(ans,r-l*2);
    ans:=min(ans,r*2-l);
  end;
  writeln(ans);
end.
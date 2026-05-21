var n:longint;
    i,j:longint;
    a,c:array[0..250,0..20]of longint;
    v:array[0..20]of boolean;
    max:int64;
procedure check;
var i,j,ans,ci:longint;
begin
  ans:=0;
  for i:=1 to n do
  begin
    ci:=0;
    for j:=1 to 10 do
    if (v[j]) then inc(ci,a[i,j]);
    inc(ans,c[i,ci]);
  end;
  if ans>max then max:=ans;
end;
procedure dfs(dep:longint;last:longint);
var i:longint;
begin
  if dep>0 then check;
  for i:=last+1 to 10 do
  begin
    v[i]:=true;
    dfs(dep+1,i);
    v[i]:=false;
  end;
end;
begin
  read(n);
  max:=-maxlongint;
  for i:=1 to n do
    for j:=1 to 10 do
    read(a[i,j]);
  for i:=1 to n do
    for j:=0 to 10 do
    read(c[i,j]);
  dfs(0,0);
  writeln(max);
end.

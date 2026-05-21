uses math;
var
  head,next,d,dfn,low,a:array[0..200005]of longint;
  insta:array[0..200005]of boolean;
  n,m,i,x,y,k,num,s,son,ans:longint;
procedure add(x,y:longint);
begin
  inc(num);
  d[num]:=y;
  next[num]:=head[x];
  head[x]:=num;
end;
procedure dfs(u,fa:longint);
var
  v,ll:longint;
  flag:boolean;
begin
  inc(s);
  dfn[u]:=s;
  low[u]:=s;
  ll:=head[u];
  insta[u]:=true;
  flag:=true;
  while ll<>0 do
  begin
    v:=d[ll];
    if v=fa then
    begin
      ll:=next[ll];
      continue;
    end;
    if dfn[v]=0 then
    begin
      dfs(v,u);
      low[u]:=min(low[u],low[v]);
      if low[v]>dfn[u] then
        inc(ans);
    end else
      low[u]:=min(low[u],dfn[v]);
    ll:=next[ll];
  end;
  insta[u]:=false;
end;
procedure qsort(l,r:longint);
var
  i,j,k,t:longint;
begin
  i:=l;
  j:=r;
  k:=(i+j) div 2;
  t:=a[k];
  a[k]:=a[i];
  while i<j do
  begin
    while (i<j) and (a[j]>t) do
      dec(j);
    if i<j then
    begin
      a[i]:=a[j];
      inc(i);
    end;
    while (i<j) and (a[i]<t) do
      inc(i);
    if i<j then
    begin
      a[j]:=a[i];
      dec(j);
    end;
  end;
  a[i]:=t;
  if i-1>l then
    qsort(l,i-1);
  if i+1<r then
    qsort(i+1,r);
end;
begin
  readln(n,m);
  for i:=1 to m do
  begin
    readln(x,y);
    add(x,y);
    add(y,x);
  end;
  for i:=1 to n do
    if dfn[i]=0 then
      dfs(i,0);
  writeln(ans);
end.
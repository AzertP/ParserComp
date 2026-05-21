type
  node=record
      y:longint;
      flow:longint;
      next:longint;
  end;
var
   first,dis,a,b,c,d:array[0..100000]of longint;
   q:array[0..1000000]of longint;
   e:array[0..20000]of node;
   i,j:longint;
   n,m:longint;
   x,y,z:longint;
   tot:longint;
   ans:longint;

procedure adde(x,y,z:longint);
begin
  e[tot].next:=first[x];
  e[tot].y:=y;
  e[tot].flow:=z;
  first[x]:=tot;
  inc(tot);
end;

procedure bfs(s:longint);
var i,now,head,tail:longint;
begin
  for i:=0 to n*2+1 do
  dis[i]:=-1;
  dis[s]:=0;
  head:=1;
  tail:=1;
  q[1]:=s;
  while head<=tail do
  begin
    now:=q[head];
    i:=first[now];
    while i<>-1 do
    begin
      y:=e[i].y;
      if (dis[y]<0)and(e[i].flow>0) then
      begin
        dis[y]:=dis[now]+1;
        inc(tail);
        q[tail]:=y;
      end;
      i:=e[i].next;
    end;
    inc(head);
  end;
end;

function dfs(x,mx:longint):longint;
var i,k:longint;
begin
  if x=n*2+1 then exit(mx);
  i:=first[x];
  while i<>-1 do
  begin
    y:=e[i].y;
    if (e[i].flow>0)and(dis[y]=dis[x]+1) then
    begin
      if e[i].flow>mx then k:=dfs(y,mx) else k:=dfs(y,e[i].flow);
      dec(e[i].flow,k);
      inc(e[i xor 1].flow,k);
      if k>0 then exit(k);
    end;
    i:=e[i].next;
  end;
  exit(0);
end;
begin
  read(n);
  for i:=0 to n*2+1 do
  first[i]:=-1;
  for i:=1 to n do
  begin
    read(a[i],b[i]);
    adde(0,i,1);
    adde(i,0,0);
  end;
  for i:=1 to n do
  begin
    read(c[i],d[i]);
    adde(n+i,n*2+1,1);
    adde(n*2+1,n+i,0);
  end;
  for i:=1 to n do
  for j:=1 to n do
  if (a[i]<c[j])and(b[i]<d[j]) then
  begin
    adde(i,n+j,1);
    adde(n+j,i,0);
  end;
  ans:=0;
  while true do
  begin
    bfs(0);
    if dis[2*n+1]<0 then break;
    repeat
      z:=dfs(0,maxint);
      inc(ans,z);
    until z>0;
  end;
  writeln(ans);
end.


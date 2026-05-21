type
  node=record
      num,id:longint;
  end;
var father,num,now,first:array[0..105000]of longint;
    p:array[0..105000]of node;
    heap:array[0..105000,0..2]of longint;
    i:longint;
    x,y:longint;
    n,m:longint;
    totin,totout:longint;
    qwq,ans,have,h,z:int64;

function getfather(x:longint):longint;
begin
  if father[x]=x then exit(x) else
  begin
    father[x]:=getfather(father[x]);
    getfather:=father[x];
  end;
end;
procedure adde(x,y:longint);
var fx,fy:longint;
begin
  fx:=getfather(x);
  fy:=getfather(y);
  if fx<>fy then
  begin
    father[fx]:=fy;
    inc(num[fy],num[fx]);
  end;
end;
procedure qs(l,r:longint);
var i,j:longint;
    t,m:node;
begin
  i:=l;
  j:=r;
  m:=p[(l+r)>>1];
  repeat
    while (p[i].id<m.id)or((p[i].id=m.id)and(p[i].num<m.num)) do inc(i);
    while (p[j].id>m.id)or((p[j].id=m.id)and(p[j].num>m.num)) do dec(j);
    if i<=j then
    begin
      t:=p[i];p[i]:=p[j];p[j]:=t;
      inc(i);
      dec(j);
    end;
  until i>j;
  if l<j then qs(l,j);
  if i<r then qs(i,r);
end;
procedure swap(var a,b:longint);
var t:longint;
begin
  t:=a;a:=b;b:=t;
end;
procedure up(x,bool:longint);
begin
  while x>1 do
  begin
    //writeln(heap[x,bool],'qwq');
    if p[now[heap[x,bool]]].num<p[now[heap[x>>1,bool]]].num then
    begin
      swap(heap[x,bool],heap[x>>1,bool]);
      x:=x>>1;
    end else break;
  end;
end;
procedure down(x,bool:longint);
var min,tot:longint;
begin
  if bool=1 then tot:=totin else tot:=totout;
  while x*2<=tot do
  begin
    min:=x*2;
    if (x*2+1<=tot)and
    (p[now[heap[x*2+1,bool]]].num<p[now[heap[min,bool]]].num) then inc(min);
    if (p[now[heap[x,bool]]].num>p[now[heap[min,bool]]].num) then
    begin
      swap(heap[x,bool],heap[min,bool]);
      x:=min;
    end else break;
  end;
end;
begin
  read(n,m);
  for i:=1 to n do
  begin
    read(p[i].num);
    father[i]:=i;
    num[i]:=1;
  end;
  for i:=1 to m do
  begin
    read(x,y);
    inc(x);
    inc(y);
    adde(x,y);
  end;
  for i:=1 to n do
  p[i].id:=getfather(i);
  qs(1,n);
  p[0].num:=maxlongint;
  i:=1;
  while i<=n do
  begin
    inc(ans,p[i].num);
    if num[p[i].id]>1 then
    begin
      now[p[i].id]:=i+1;
      inc(have,num[p[i].id]);
      inc(h);
    end else
    begin
      now[p[i].id]:=0;
      inc(z);
    end;
    inc(totout);
    heap[totout,0]:=p[i].id;
    up(totout,0);
    first[p[i].id]:=i;
    i:=i+num[p[i].id];
  end;
  if totout=1 then
  begin
    writeln(0);
    exit;
  end;
  have:=have-(h-1)*2;
  if have<z then
  begin
    writeln('Impossible');
    exit;
  end;
  while totout>0 do
  begin
    inc(qwq);
    x:=heap[1,0];
    if now[x]<>0 then
    begin
      inc(totin);
      heap[totin,1]:=x;
      up(totin,1);
    end;
    if (qwq>2)and(totin>0) then
    begin
      x:=heap[1,1];
      inc(ans,p[now[x]].num);
      inc(now[x]);
      if (now[x]>=first[x]+num[x]) then
      begin
        now[x]:=0;
        heap[1,1]:=heap[totin,1];
        dec(totin);
      end;
      if totin>0 then down(1,1);
    end;
    heap[1,0]:=heap[totout,0];
    dec(totout);
    down(1,0);
  end;
  writeln(ans);
end.

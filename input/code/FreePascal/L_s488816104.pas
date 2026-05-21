var
  a,f,g:array[0..100000]of longint;
  p1,p2,q1,q2:array[0..100000]of longint;
  i,j,n,ans,n1,n2:longint;
Procedure max(x:longint);
begin
  if x>ans then ans:=x;
end;
Procedure ch(var x,y:longint);
var
  t:longint;
begin
  t:=x;x:=y;y:=t;
end;
Procedure qs1(l,r:longint);
var
  i,j,mid:longint;
begin
  i:=l;j:=r;mid:=p1[(l+r)div 2];
  repeat
    while p1[i]>mid do inc(i);
    while p1[j]<mid do dec(j);
    if i<=j then
      begin
        ch(p1[i],p1[j]);
        ch(p2[i],p2[j]);
        inc(i);dec(j);
      end;
  until i>j;
  if l<j then qs1(l,j);
  if i<r then qs1(i,r);
end;
Procedure qs2(l,r:longint);
var
  i,j,mid:longint;
begin
  i:=l;j:=r;mid:=q1[(l+r)div 2];
  repeat
    while q1[i]>mid do inc(i);
    while q1[j]<mid do dec(j);
    if i<=j then
      begin
        ch(q1[i],q1[j]);
        ch(q2[i],q2[j]);
        inc(i);dec(j);
      end;
  until i>j;
  if l<j then qs2(l,j);
  if i<r then qs2(i,r);
end;
begin
  readln(n);
  for i:=1 to n do
    read(a[i]);
  for i:=1 to n do
    begin
      if i mod 2=1 then inc(f[a[i]]) else inc(g[a[i]]);
    end;
  for i:=0 to 100000 do
    begin
      if f[i]<>0 then
        begin
          inc(n1);
          p1[n1]:=f[i];
          p2[n1]:=i;
        end;
      if g[i]<>0 then
        begin
          inc(n2);
          q1[n2]:=g[i];
          q2[n2]:=i;
        end;
    end;
  qs1(1,n1);
  qs2(1,n2);
  for i:=1 to n1 do
    begin
      if p2[i]<>q2[1] then max(p1[i]+q1[1]) else max(p1[i]+q1[2]);
    end;
  write(n-ans);
end.
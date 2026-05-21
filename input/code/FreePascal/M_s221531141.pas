program V1501;
 type
  w=array[0..100001] of int64;
 var
  a,b,c,d,e,f:w;
  i,n,p:longint;
 procedure haha(var a:w;l,r:longint);
  var
   o,p,m,t:longint;
  begin
   o:=l;
   p:=r;
   m:=a[(l+r) div 2];
   repeat
    while a[o]<m do inc(o);
    while a[p]>m do dec(p);
    if o<=p then
     begin
      t:=a[o];
      a[o]:=a[p];
      a[p]:=t;
      inc(o);
      dec(p);
     end;
   until o>p;
   if o<r then haha(a,o,r);
   if l<p then haha(a,l,p);
  end;
 begin
  readln(n);
  for i:=1 to n do
   read(a[i]);
  readln;
  a[n+1]:=1008208820;
  haha(a,1,n+1);
  for i:=1 to n do
   read(b[i]);
  readln;
  b[n+1]:=1008208820;
  haha(b,1,n+1);
  for i:=1 to n do
   read(c[i]);
  readln;
  c[n+1]:=1008208820;
  haha(c,1,n+1);
  for i:=1 to n+1 do
   f[i]:=n-i+1;
  p:=1;
  for i:=1 to n do
   begin
    while c[p]<=b[i] do inc(p);
    e[i]:=f[p];
   end;
  e[n+1]:=0;
  for i:=n downto 1 do
   inc(e[i],e[i+1]);
  p:=1;
  for i:=1 to n do
   begin
    while b[p]<=a[i] do inc(p);
    d[i]:=e[p];
   end;
  d[n+1]:=0;
  for i:=n downto 1 do
   inc(d[i],d[i+1]);
  writeln(d[1]);
 end.

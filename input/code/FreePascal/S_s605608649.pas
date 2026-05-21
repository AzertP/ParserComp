program V1004;
 var
  a:array[0..200001,0..1] of longint;
  i,m,n,x,y,u:longint;
 begin
  readln(n,m);
  fillchar(a,sizeof(a),0);
  a[1,0]:=1;
  a[n,1]:=1;
  for i:=1 to m do
   begin
    readln(x,y);
    if x=1 then inc(a[y,0]);
    if y=n then inc(a[x,1]);
   end;
  u:=0;
  for i:=1 to n do
   if (a[i,0]>0) and (a[i,1]>0) then u:=1;
  if u=1 then writeln('POSSIBLE')
         else writeln('IMPOSSIBLE');
 end.

var
  a,b:array[0..1000]of longint;
  n,m,i,j,m1,m2,x,y:longint;
begin
   readln(n,m,x,y);
   m1:=x;m2:=y;
   for i:=1 to n do
     begin
       read(a[i]);
       if a[i]>m1 then m1:=a[i];
     end;
   readln;
   for i:=1 to m do
     begin
       read(b[i]);
       if b[i]<m2 then m2:=b[i];
     end;

   if m2<=m1 then write('War') else write('No War');
end.
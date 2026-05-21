var
 i,j,n,l:longint;
 a:array[1..100] of string;
 t:string;
begin
 readln(n,l);
 for i:=1 to n do readln(a[i]);
 for i:=1 to n-1 do 
 for j:=i+1 to n do 
 if a[i]>a[j] then
  begin
   t:=a[i];
   a[i]:=a[j];
   a[j]:=t;
  end;
  for i:=1 to n do write(a[i]);
  end.
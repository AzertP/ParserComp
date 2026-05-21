var
  n,m,i,k:qword;
  sum:longint;
begin
  readln(n,m);
  k:=m div n;
  i:=1;
  while i<=k do
  begin
   i:=i*2;
   sum:=sum+1;
   end;
   write(sum);
end.

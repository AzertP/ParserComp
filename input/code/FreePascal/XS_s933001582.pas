var n,m,i,cnt,sum  :int64;
    a:array[1..1000]of int64;
begin
  readln(n,m);
  for i:=1 to n do read(a[i]);
  sum:=0; cnt:=0;
  for i:=1 to n do sum:=sum+a[i];
  for i:=1 to n do begin
    if a[i]*4*m>=sum then cnt:=cnt+1;
  end;
  if cnt>=m then writeln('Yes')
  else writeln('No');  
end. 
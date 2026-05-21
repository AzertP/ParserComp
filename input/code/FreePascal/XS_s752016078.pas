var
  c,t,m,i,j,minc,mint,n:longint;
begin
  readln(n,m);
  minc:=10000;
  for i:=1 to n do
    begin
      readln(c,t);
      if t<=m then
        if c<minc then minc:=c;
    end;
  if minc=10000 then write('TLE') else write(minc);
end.
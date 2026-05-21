VAR N,i,j,count,kq:INTEGER;
begin
 readln(n);
 i:=1;
 if n < 105 then writeln('0') else
 if n = 105 then writeln('1') else 
 begin
   while i <= n do 
    begin
    count:=0;
    for j:=1 to n do 
     if i mod j = 0 then inc(Count);
    if count = 8 then inc(kq);
    inc(i,2);
    end;
  writeln(kq);
 end;
 end.
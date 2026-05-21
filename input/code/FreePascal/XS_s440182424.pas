var
 a,b,t:int64;
begin
 readln(a,b);
 if a>b then
  begin
   t:=a; a:=b; b:=t;
  end;
 if (a<=0) and (b>=0) then writeln('Zero')
  else if (a<0) and (b<0) and (abs(a-b) mod 2=0) then writeln('Negative')
   else writeln('Positive');
end.
var
 i,x,s,sum:longint;
begin
 for i:=1 to 3 do
  begin
   read(x);
   if x=5 then inc(s);
   if x=7 then inc(sum);
  end;
 if (s=2) and (sum=1) then writeln('YES')
                           else writeln('NO');
end.
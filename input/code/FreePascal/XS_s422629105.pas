var ans:int64;
    q,h,s,d,n:int64;
begin
  read(q,h,s,d);
  if 4*q<s then s:=4*q;
  if 2*h<s then s:=2*h;
  if 2*s<d then d:=2*s;
  read(n);
  if s*2>d then inc(ans,(n div 2)*d) else inc(ans,s*n);
  if (n mod 2=1)and(s*2>d) then inc(ans,s);
  writeln(ans);
end.
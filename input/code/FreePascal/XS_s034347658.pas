var n,l,r,ans,i:longint;
begin
  readln(n);
  for i:=1 to n do
  begin 
    readln(l,r);
    inc(ans,r-l+1);
  end;
  writeln(ans);
end.
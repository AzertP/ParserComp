var
  a,b,x,ai,bi:int64;
begin
  readln(a,b,x);
  if a mod x=0 then ai:=a div x
               else ai:=a div x+1;
  bi:=b div x;
  writeln(bi-ai+1);
end.
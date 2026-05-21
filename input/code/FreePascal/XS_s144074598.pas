program main;
var n : integer;

begin
  readln(n);
  if n mod 1000 = 0 then
     writeln(n mod 1000)
  else
     writeln((n div 1000 + 1) * 1000 - n);
end.
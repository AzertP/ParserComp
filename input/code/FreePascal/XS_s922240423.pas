var i,j:longint;
    s:ansistring;
begin
  readln(s);
  for i:=1 to length(s) do
    if s[i]='C' then break;
  for j:=i+1 to length(s) do
    if s[j]='F' then
    begin
      writeln('Yes');
      halt;
    end;
  writeln('No');
end.
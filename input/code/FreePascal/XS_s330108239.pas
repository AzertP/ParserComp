var
 s,sx:ansistring;
 i:longint;
begin
 readln(s);
 for i:=1 to length(s) do
  if s[i]='1' then sx:=sx+'1'
   else if s[i]='0' then sx:=sx+'0'
    else if s[i]='B' then 
     if sx<>'' then delete(sx,length(sx),1);
 writeln(sx);
end.
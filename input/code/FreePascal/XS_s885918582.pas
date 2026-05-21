var
 s1,s2:ansistring;
 i:longint;
begin
 readln(s1);
 for i:=1 to length(s1) do
  case s1[i] of
   '0':s2:=s2+'0';
   '1':s2:=s2+'1';
   'B':delete(s2,length(s2),1);
  end;
  writeln(s2);
end.
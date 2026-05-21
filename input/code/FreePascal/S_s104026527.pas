var
 i,n,s,max:longint;
 ch:char;
begin
 readln(n);
 s:=0;
 for i:=1 to n do
  begin
   read(ch);
   case ch of
    'I':inc(s);
    'D':dec(s);
   end;
   if s>max then max:=s;
  end;
 writeln(max);
end.
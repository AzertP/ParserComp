var w,n,e,s:longint;
    x:char;
begin
  while not eoln do
  begin
    read(x);
    case x of
      'W':w:=1;
      'N':n:=1;
      'E':e:=1;
      'S':s:=1;
    end;
  end;
  if (w=e) and (n=s) then
    writeln('Yes')
  else
    writeln('No');
end.
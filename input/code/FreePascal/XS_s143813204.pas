var
  n:string;
begin
  readln(n);
  if n[1]=n[length(n)] then
    begin
      if odd(length(n)) then writeln('Second')
        else writeln('First');
    end
    else
      begin
        if odd(length(n)) then writeln('First')
         else writeln('Second');
      end;
end.
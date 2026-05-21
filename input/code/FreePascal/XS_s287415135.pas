var
  a,b:longint;
begin
  readln(a,b);
  if (a*b)mod 2=1 then writeln('Odd')
    else writeln('Even');
  close(input);
  close(output);
end.
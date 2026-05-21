var i, n, k, S, v:Longint;
begin
  S := 0;
  read(k, n);
  for i := 1 to n do
  begin
    read(v);
    S := S + v;
    if S >= k then
      begin
        writeln('Yes');
        exit;
      end;
  end;
  writeln('No');
end.

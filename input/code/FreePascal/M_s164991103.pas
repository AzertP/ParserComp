var

  s : ansistring;
  i, cnt : longint;
  b : boolean;
  
begin

  readln(S);
  b := true;
  cnt := 0;
  
  for i := 1 to length(s)-1 do
  begin
    if s[i] <> s[i+1] then
      b := false;
  end;
  
  if b = true then
  begin
    writeln(0);
    halt;
  end
  else
    for i := 1 to length(s)-1 do
      if ((s[i] = 'B') and (s[i+1] = 'W')) or ((s[i] = 'W') and (s[i+1] = 'B')) then
        inc(cnt);
        
  writeln(cnt);
  

end.
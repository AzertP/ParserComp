var a:array[0..101,0..101] of longint;
    i,j,n,m:longint;
    ch:char;
begin
  readln(n,m);
  for i:=1 to n do
  begin
    for j:=1 to m do
    begin
      read(ch);
      if ch='#' then
      begin
        if a[i-1,j-1]<>-1 then inc(a[i-1,j-1]);
        if a[i-1,j]<>-1 then inc(a[i-1,j]);
        if a[i-1,j+1]<>-1 then inc(a[i-1,j+1]);
        if a[i,j-1]<>-1 then inc(a[i,j-1]);
        a[i,j]:=-1;
        if a[i,j+1]<>-1 then inc(a[i,j+1]);
        if a[i+1,j-1]<>-1 then inc(a[i+1,j-1]);
        if a[i+1,j]<>-1 then inc(a[i+1,j]);
        if a[i+1,j+1]<>-1 then inc(a[i+1,j+1]);
      end;
    end;
    readln;
  end;
  for i:=1 to n do
  begin
    for j:=1 to m do
    begin
      if a[i,j]=-1 then write('#')
      else write(a[i,j]);
    end;
    writeln;
  end;
end.
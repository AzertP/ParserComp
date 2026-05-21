var
a:array[0..55,0..55] of char;
n,m,i,j,s:longint;
begin
readln(n,m);
for i:=1 to n do
  begin
    for j:=1 to m do read(a[i,j]);
    readln;
  end;
for i:=0 to m+1 do begin a[0,i]:=' ';a[n+1,i]:=' ';end;
for i:=1 to n do begin a[i,0]:=' ';a[i,m+1]:=' ';end;
for i:=1 to n do
  begin
    for j:=1 to m do
      if a[i,j]='.' then
        begin
          s:=0;
          if a[i-1,j-1]='#' then inc(s);
          if a[i-1,j]='#' then inc(s);
          if a[i-1,j+1]='#' then inc(s);
          if a[i,j-1]='#' then inc(s);
          if a[i,j+1]='#' then inc(s);
          if a[i+1,j-1]='#' then inc(s);
          if a[i+1,j]='#' then inc(s);
          if a[i+1,j+1]='#' then inc(s);
          write(s);
        end
        else write(a[i,j]);
    writeln;
  end;
end.
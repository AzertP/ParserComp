var
  a, f, h, tmp: array [0..100000] of longint;
  u, v: array [1..200000] of longint;
  n, m, i, x, y, s, rt, re, cnt: longint;
procedure swap(var x, y: longint); inline;
var
  t: longint;
begin
  t := x;
  x := y;
  y := t;
end;
function find(x: longint): longint;
begin
  if x <> f[x] then f[x] := find(f[x]);
  exit(f[x]);
end;
procedure add(x, y: longint); inline;
begin
  inc(cnt);
  u[cnt] := h[x];
  v[cnt] := y;
  h[x] := cnt;
end;
procedure dfs(x, s: longint);
var
  i: longint;
begin
  a[x] := s;
  i := h[x];
  while i > 0 do
  begin
    if v[i] <> f[x] then
    begin
      f[v[i]] := x;
      dfs(v[i], -s);
      inc(a[x], a[v[i]]);
    end;
    i := u[i];
  end;
end;
procedure quicksort(l, r: longint);
var
  i, j, m: longint;
begin
  i := l;
  j := r;
  m := tmp[(l + r) shr 1];
  repeat
    while tmp[i] < m do inc(i);
    while tmp[j] > m do dec(j);
    if i <= j then
    begin
      swap(tmp[i], tmp[j]);
      inc(i);
      dec(j);
    end;
  until i > j;
  if l < j then quicksort(l, j);
  if i < r then quicksort(i, r);
end;
begin
  read(n, m);
  for i := 1 to n do f[i] := i;
  cnt := 0;
  rt := 1;
  re := 1;
  for i := 1 to m do
  begin
    read(x, y);
    if find(x) <> find(y) then
    begin
      f[find(x)] := find(y);
      add(x, y);
      add(y, x);
    end
    else
    begin
      rt := x;
      re := y;
    end;
  end;
  f[rt] := 0;
  dfs(rt, 1);
  if n = m then
  begin
    cnt := 0;
    i := re;
    while i > 0 do
    begin
      inc(cnt);
      tmp[cnt] := a[i];
      i := f[i];
    end;
    if odd(cnt) then
    begin
      if odd(a[rt]) then
      begin
        writeln(-1);
        halt;
      end;
      i := re;
      while i > 0 do
      begin
        dec(a[i], a[rt] div 2);
        i := f[i];
      end;
    end
    else
    begin
      if a[rt] <> 0 then
      begin
        writeln('-1');
        halt;
      end;
      quicksort(1, cnt);
      i := re;
      while i > 0 do
      begin
        dec(a[i], tmp[cnt shr 1]);
        i := f[i];
      end;
    end;
  end
  else
    if a[rt] <> 0 then
    begin
      writeln(-1);
      halt;
    end;
  s := 0;
  for i := 1 to n do inc(s, abs(a[i]));
  writeln(s);
end.

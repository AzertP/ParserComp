var
  a, h, f, w, id, hs, sum: array [0..200000] of longint;
  u, v: array [0..400000] of longint;
  s, t: array [1..800000] of longint;
  n, q, i, x, y, num: longint;
procedure swap(var x, y: longint); inline;
var
  t: longint;
begin
  t := x;
  x := y;
  y := t;
end;
procedure add(x, y: longint); inline;
begin
  inc(num);
  u[num] := h[x];
  v[num] := y;
  h[x] := num;
end;
procedure pushup(x: longint); inline;
begin
  s[x] := s[x shl 1] + s[x shl 1 or 1];
end;
procedure pushdown(x, l, r: longint); inline;
var
  m: longint;
begin
  m := (l + r) shr 1;
  inc(s[x shl 1], t[x] * (m - l + 1));
  inc(s[x shl 1 or 1], t[x] * (r - m));
  inc(t[x shl 1], t[x]);
  inc(t[x shl 1 or 1], t[x]);
  t[x] := 0;
end;
procedure build(x, l, r: longint);
var
  m: longint;
begin
  if l = r then
  begin
    s[x] := w[l];
    exit;
  end;
  m := (l + r) shr 1;
  build(x shl 1, l, m);
  build(x shl 1 or 1, m + 1, r);
  pushup(x);
end;
procedure update(x, l, r, tl, tr, k: longint);
var
  m: longint;
begin
  if (l >= tl) and (r <= tr) then
  begin
    inc(s[x], k * (r - l + 1));
    inc(t[x], k);
    exit;
  end;
  pushdown(x, l, r);
  m := (l + r) shr 1;
  if tl <= m then
    update(x shl 1, l, m, tl, tr, k);
  if tr > m then
    update(x shl 1 or 1, m + 1, r, tl, tr, k);
  pushup(x);
end;
function query(x, l, r, t: longint): longint;
var
  m: longint;
begin
  if l = r then  
    exit(s[x]);
  pushdown(x, l, r);
  m := (l + r) shr 1;
  if t <= m then
    exit(query(x shl 1, l, m, t))
  else
    exit(query(x shl 1 or 1, m + 1, r, t));
end;
procedure dfs1(x, y, z: longint);
var
  i, max: longint;
begin
  f[x] := y;
  sum[x] := 1;
  hs[x] := 0;
  max := 0;
  i := h[x];
  while i > 0 do
  begin
    if v[i] <> y then
    begin
      dfs1(v[i], x, z + 1);
      if sum[v[i]] > max then
      begin
        max := sum[v[i]];
        hs[x] := v[i];
      end;
      inc(sum[x], sum[v[i]]);
    end;
    i := u[i];
  end;
end;
procedure dfs2(x, y: longint);
var
  i: longint;
begin
  inc(num);
  id[x] := num;
  w[id[x]] := a[x];
  if hs[x] = 0 then
    exit;
  dfs2(hs[x], y);
  i := h[x];
  while i > 0 do
  begin
    if (v[i] <> f[x]) and (v[i] <> hs[x]) then
      dfs2(v[i], v[i]);
    i := u[i];
  end;
end;
procedure updatesubtree(x, k: longint); inline;
begin
  update(1, 1, n, id[x], id[x] + sum[x] - 1, k);
end;
begin
  read(n, q);
  num := 0;
  for i := 1 to n - 1 do
  begin
    read(x, y);
    add(x, y);
    add(y, x);
  end;
  num := 0;
  dfs1(1, 0, 0);
  dfs2(1, 1);
  build(1, 1, n);
  for i := 1 to q do
  begin
    read(x, y);
    updatesubtree(x, y);
  end;
  for i := 1 to n do
    write(query(1, 1, n, id[i]), ' ');
end.

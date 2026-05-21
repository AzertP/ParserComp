var
  a: array [1..50] of real;
  n, i, x, num: longint;
procedure swap(var x, y: real); inline;
var
  t: real;
begin
  t := x;
  x := y;
  y := t;
end;
procedure push(x: real); inline;
var
  i: longint;
begin
  inc(num);
  a[num] := x;
  i := num;
  while (i > 1) and (a[i] < a[i shr 1]) do
  begin
    swap(a[i], a[i shr 1]);
    i := i shr 1;
  end;
end;
function pop: real; inline;
var
  i, x: longint;
begin
  pop := a[1];
  a[1] := a[num];
  dec(num);
  i := 1;
  x := 2;
  if (x < num) and (a[x + 1] < a[x]) then
    inc(x);
  while (x <= num) and (a[x] < a[i]) do
  begin
    swap(a[x], a[i]);
    i := x;
    x := i shl 1;
    if (x < n) and (a[x + 1] < a[x]) then
      inc(x);
  end;
end;
begin
  read(n);
  num := 0;
  for i := 1 to n do
  begin
    read(x);
    push(x);
  end;
  for i := 1 to n - 1 do
    push((pop + pop) / 2);
  writeln(pop : 0 : 6);
end.

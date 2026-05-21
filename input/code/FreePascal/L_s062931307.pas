program reconciled;
uses math;
const
  md: int64 = 1000000007;
  f10000: int64 = 531950728;
  f20000: int64 = 368774859;
  f30000: int64 = 548996970;
  f40000: int64 = 422550956;
  f50000: int64 = 737935835;
  f60000: int64 = 309944332;
  f70000: int64 = 296716438;
  f80000: int64 = 65533322;
  f90000: int64 = 851076783;
  f100000: int64 = 457992974;
function fact(n: int64): int64;
var
  i, k, f: int64;
begin
  f := f100000;
  k := 100001;
  if n < 100000 then
  begin
    f := f90000;
    k := 90001;
  end;
  if n < 90000 then
  begin
    f := f80000;
    k := 80001;
  end;
  if n < 80000 then
  begin
    f := f70000;
    k := 70001;
  end;
  if n < 60000 then
  begin
    f := f50000;
    k := 50001;
  end;
  if n < 50000 then
  begin
    f := f40000;
    k := 40001;
  end;
  if n < 40000 then
  begin
    f := f30000;
    k := 30001;
  end;
  if n < 30000 then
  begin
    f := f20000;
    k := 20001;
  end;
  if n < 20000 then
  begin
    f := f10000;
    k := 10001;
  end;
  if n < 10000 then
  begin
    f := 1;
    k := 1;
  end;
  for i := k to n do
    f := i*f mod md;
  fact := f;
end;
var
  n, m, x: int64;
begin
  read(n, m);
  if abs(n-m) > 1 then
  begin
    writeln(0);
    halt(0);
  end;
  x := ifthen(n = m, 2, 1);
  x := x*fact(n) mod md;
  x := x*fact(m) mod md;
  writeln(x);
end.
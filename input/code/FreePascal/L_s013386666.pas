{$R+,S+,Q+,I+,O-}
{R-,S-,Q-,I-,O+}
// Code By H~$~C

uses math;

type
  int32 = int64;
  double = extended;
  bool = boolean;

const
  inf = 1061109567;
  lnf = 4557430888798830399;
  Maxn = 200005;
  LOG = 32;

var
  i, j: int32;
  swap_tmp, x, y, z: int64;
  n, T, sza, szb, ans: int64;
  a: array [0 .. Maxn, 0 .. 1] of int64;
  b: array [0 .. Maxn] of int64;
  dp: array [0 .. LOG] of int64;

function qsort_less(i, j: int32): bool;
begin
  qsort_less := (a[j][0] * (a[i][1] + 1) < (a[i][0] * (a[j][1] + 1)));
end;
procedure qsort_swap(i, j: int32);
begin
  a[0] := a[i];
  a[i] := a[j];
  a[j] := a[0];
end;
procedure qsort(_l, _r: int32);
  var i, j, _p: int32;
begin
  if (_r = _l) then exit;
  i := _l; j := _r;
  _p := _l;
  repeat
    while qsort_less(i, _p) do inc(i);
    while qsort_less(_p, j) do dec(j);
    if (i <= j) then begin
      qsort_swap(i, j);
      inc(i); dec(j);
    end;
  until (i > j);
  if (i < _r) then qsort(i, _r);
  if (j > _l) then qsort(_l, j);
end;

function qsort2_less(i, j: int32): bool;
begin
  qsort2_less := (b[i] < b[j]);
end;
procedure qsort2_swap(i, j: int32);
begin
  b[0] := b[i];
  b[i] := b[j];
  b[j] := b[0];
end;
procedure qsort2(_l, _r: int32);
  var i, j, _p: int32;
begin
  if (_r = _l) then exit;
  i := _l; j := _r;
  _p := _l;
  repeat
    while qsort2_less(i, _p) do inc(i);
    while qsort2_less(_p, j) do dec(j);
    if (i <= j) then begin
      qsort2_swap(i, j);
      inc(i); dec(j);
    end;
  until (i > j);
  if (i < _r) then qsort2(i, _r);
  if (j > _l) then qsort2(_l, j);
end;

begin
//  assign(input,'in'); reset(input);
//  assign(output,'out'); rewrite(output);
  randomize;
  
  readln(n, T);
  sza := 0; szb := 0;
  for i := 1 to n do begin
    readln(x, y);
    if (x = 0) then begin
      inc(szb);
      b[szb] := y;
    end
    else begin
      inc(sza);
      a[sza, 0] := x;
      a[sza, 1] := y;
    end;
  end;
  
  for i := 2 to sza do begin
    x := random(i - 1) + 1;
    a[0] := a[x];
    a[x] := a[i];
    a[i] := a[0];
  end;
  qsort(1, sza);
  
  for i := 1 to LOG do dp[i] := T + 1;
  dp[0] := 0;
  for i := 1 to sza do begin
    for j := LOG - 1 downto 0 do begin
      if (dp[j] > T) then continue;
      if (dp[j + 1] > dp[j] + 1 + (dp[j] + 1) * a[i][0] + a[i][1]) then begin
        dp[j + 1] := dp[j] + 1 + (dp[j] + 1) * a[i][0] + a[i][1];
      end;
    end;
  end;
  
  for i := 2 to szb do begin
    x := random(i - 1) + 1;
    b[0] := b[x];
    b[x] := b[i];
    b[i] := b[0];
  end;
  qsort2(1, szb);
  ans := 0;
  for i := 0 to LOG do begin
    if (dp[i] > T) then continue;
    x := T - dp[i];
    y := 0;
    for j := 1 to szb do begin
      if (x > b[j]) then begin
        dec(x, b[j] + 1);
        inc(y);
      end
      else break;
    end;
    if (i + y > ans) then ans := i + y;
  end;
  
  writeln(ans);
  
  close(input);
  close(output);
end.

{$R+,S+,Q+,I+,O-}
{R-,S-,Q-,I-,O+}
// Code By H~$~C

uses math;

type
  int32 = longint;
  double = extended;
  bool = boolean;

const inf = 1061109567;

var
  i, j, swap_tmp, x, y, z: int32;
  n, ans: int32;
  h, p: array [0 .. 5005] of int32;
  dp: array [0 .. 5005] of int64;

function qsort_less(i, j: int32): bool;
begin
  qsort_less := (h[i] + p[i]) < (h[j] + p[j]);
end;
procedure qsort_swap(i, j: int32);
  var _swap_tmp: int32;
begin
  _swap_tmp := h[i]; h[i] := h[j]; h[j] := _swap_tmp;
  _swap_tmp := p[i]; p[i] := p[j]; p[j] := _swap_tmp;
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

begin
//  assign(input,'in'); reset(input);
//  assign(output,'1.out'); rewrite(output);
  randseed := 6714;
  
  readln(n);
  for i := 1 to n do begin
    readln(h[i], p[i]);
  end;
  qsort(1, n);
  
  fillchar(dp, sizeof(dp), 63);
  dp[0] := 0;
  for i := 1 to n do begin
    for j := i - 1 downto 0 do begin
      if (dp[j] <= h[i]) and (dp[j + 1] > dp[j] + p[i]) then begin
        dp[j + 1] := dp[j] + p[i];
      end;
    end;
  end;
  
  ans := 0;
  for i := 0 to n do begin
    if (dp[i] < 1000000000000000000) then begin
      ans := i;
    end;
  end;
  writeln(ans);
  
  close(input);
  close(output);
end.

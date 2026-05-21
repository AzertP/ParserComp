{$mode objfpc}
program D;
uses
  fgl, math;

type
  TIntLIst = specialize TFPGList<Int64>;

var
  N: Integer;
  C: Int64;
  xi, li, ri, vi: TIntLIst;

procedure ReadData;
var
  i: Integer;
  x, v: Int64;

begin
  ReadLn(N, C);

  li := TIntLIst.Create;
  ri := TIntLIst.Create;
  ri.Count := N;
  xi := TIntLIst.Create;
  vi := TIntLIst.Create;

  for i := 1 to N do
  begin
    ReadLn(x, v);
    xi.Add(x);
    li.Add(x);
    vi.Add(v);
  end;

  ri.Count := N;
  for i := N - 1 downto 0 do
    ri[i] := c - xi[i];
end;

var
  L1, L2, R1, R2: TIntList;
  L1M, L2M, R1M, R2M: TIntList;

function Solve: Int64;
var
  i: Integer;
  l, r: Integer;
  Current: Int64;
  CurSum: Int64;

begin
  L1 := TIntLIst.Create;
  L2 := TIntLIst.Create;
  R1 := TIntLIst.Create;
  R2 := TIntLIst.Create;
  L1M := TIntLIst.Create;
  L2M := TIntLIst.Create;
  R1M := TIntLIst.Create;
  R2M := TIntLIst.Create;
  L1.Count := N + 1;
  L2.Count := N + 1;
  R1.Count := N + 1;
  R2.Count := N + 1;
  L1M.Count := N + 1;
  L2M.Count := N + 1;
  R1M.Count := N + 1;
  R2M.Count := N + 1;

{
l1(l) := v0+...+vl-(xl)
l2(l) := v0+...+vr-2(xl)
r2(r) := vr+...+vN-1-(C - xr)
r2(r) := vr+...+vN-1-2(C - xr)
}
  l := 0;
  L1[l] := vi[l] - xi[l];
  L2[l] := vi[l] - 2 * xi[l];
  L1M[l] := L1[l];
  L2M[l] := L2[l];
  CurSum := vi[l];
  for l := 1 to N - 1 do
  begin
    Inc(CurSum, vi[l]);
    L1[l] := CurSum - xi[l];
    L2[l] := CurSum - 2 * li[l];
    L1M[l] := Max(L1[l], L1M[l - 1]);
    L2M[l] := Max(L2[l], L2M[l - 1]);
  end;

  r := N - 1;
  CurSum := vi[r];
  R1[r] := CurSum - (C - xi[r]);
  R2[r] := CurSum - 2 * (C - xi[r]);
  R1M[r] := R1[r];
  R2M[r] := R2[r];
  for r := N - 2 downto 0 do
  begin
    Inc(CurSum, vi[r]);
    R1[r] := CurSum - ri[r];
    R2[r] := CurSum - 2 * ri[r];
    R1M[r] := Max(R1[r], R1M[r + 1]);
    R2M[r] := Max(R2[r], R2M[r + 1]);
  end;

  Result := 0;

  for i := 0 to N - 2 do
  begin
    if Result < L1[i] then
      Result := L1[i];
    if Result < R1[i] then
      Result := R1[i];

    l := i;
    r := (i + 1);
  {
    WriteLn('L1:', L1[l], ' L2:', L2[l]); 
    WriteLn('R1:', R1[r], ' R2:', R2[r]); 
    WriteLn('LM:', L1M[l], ' L2M:', L2M[l]); 
    WriteLn('RM:', R1M[r], ' R2M:', R2M[r]); 
  }

    if Result < L2[l] + R1M[r] then
      Result := L2[l] + R1M[r];

    if Result < L1M[l] + R2[r] then
      Result := L1M[l] + R2[r];
  end;
  i := N - 1;
  if Result < L1[i] then
    Result := L1[i];
  if Result < R1[i] then
    Result := R1[i];
end;

begin
  ReadData;

  WriteLn(Solve);
end.
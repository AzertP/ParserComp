uses math;
const fi='knapsack2.inp';
      fo='knapsack2.out';
      maxn=100;
      maxv=200000;
var n,w:int32;
    maxval:uint64;
    we,v:array[1..maxn] of uint32;
    f:array[0..maxn,0..maxv] of uint32;
//=============================
procedure nhap;
var i:int32;
begin
   readln(n,w); maxval:=0;
   for i:=1 to n do
   begin
     readln(we[i],v[i]);
     maxval:=maxval+v[i];
   end;
end;
//=============================
function min(x,y:int64):int64;
begin
   if x>y then exit(y)
   else exit(x);
end;
//=============================
procedure xuli;
var i,j:int32;
begin
   for i:=0 to n do
    for j:=0 to maxval do f[i,j]:=maxlongint;
   f[0,0]:=0;
   for i:=0 to n-1 do
    for j:=0 to maxval do
     if f[i,j]<>maxlongint then
      begin
         f[i+1,j]:=min(f[i+1,j],f[i,j]);
         if f[i,j]+we[i+1]<=w then
         f[i+1,j+v[i+1]]:=min(f[i+1,j+v[i+1]],f[i,j]+we[i+1]);
      end;
   for i:=maxval downto 0 do
    if f[n,i]<=w then break;
   write(i);
end;
//=============================
begin
//   assign(input,fi); reset(input);
//   assign(output,fo); rewrite(output);
   nhap;
   xuli;
//   close(input); close(output);
end.

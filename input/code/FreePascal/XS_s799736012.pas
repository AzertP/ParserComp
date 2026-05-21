var n,m,i,j,ans,l:longint;
var a:array[0..1020] of longint;
begin
  read(n);ans:=0;l:=101;
  for i:=1 to n do
  begin
    read(a[i]);
    if (a[i]=a[i-1])then begin a[i]:=l;inc(ans);inc(l);end;
  end;
  writeln(ans);
end.
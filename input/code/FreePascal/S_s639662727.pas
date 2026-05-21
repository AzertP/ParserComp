program aaa;
var
  n,i:longint;
  ans:int64;
  a:array[0..100010]of longint;
begin
  readln(n);
  a[0]:=0;
  for i:=1 to n do
  begin
    read(a[i]);
    ans:=ans+abs(a[i]-a[i-1]);
  end;
  a[n+1]:=0;
  ans:=ans+abs(a[n]);
  for i:=1 to n do
   writeln(ans+abs(a[i+1]-a[i-1])-abs(a[i]-a[i-1])-abs(a[i+1]-a[i]));
  readln;
  readln;
end.     
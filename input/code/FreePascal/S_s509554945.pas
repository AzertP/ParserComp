var  now,ans,ansx:longint;
     x,a:array[0..150000]of longint;
     n:longint;
     i:longint;
begin
  read(n);
  now:=0;
  for i:=1 to n do
  begin
    read(a[i]);
    x[i]:=now;
    ans:=ans+abs(now-a[i]);
    now:=a[i];
  end;
  a[n+1]:=0;
  x[n+1]:=now;
  ans:=ans+abs(now-a[i+1]);
  for i:=1 to n do
  begin
    ansx:=ans-abs(x[i]-a[i])-abs(x[i+1]-a[i+1])+abs(x[i]-a[i+1]);
    writeln(ansx);
  end;
end.
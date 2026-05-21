var i,j,k,m,n,t,ok,tt:longint;
a,tar:real;
ar:array [0..110] of longint;
begin
  readln(n,m);
  for i:=1 to n do
  begin
    read(ar[i]);
    tt:=tt+ar[i];
  end;  
  for i:=1 to n do
    for j:=1 to n-1 do
      if ar[j]>ar[j+1] then
      begin
        t:=ar[j];
        ar[j]:=ar[j+1];
        ar[j+1]:=t;
      end;   
  tar:=(1/(4*m));
  for i:=n downto (n-m)+1 do
  begin
    a:=(ar[i]/tt);
    if a>=tar then
      inc(ok);
  end;    
  if ok=m then  
    writeln('Yes')
  else 
    writeln('No'); 
end.  
      
      
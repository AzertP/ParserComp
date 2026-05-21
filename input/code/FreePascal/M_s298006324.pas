var
  n,y,i,j,k:longint;
begin
  readln(n,y);
  if (y div n=10000) then
  begin
     writeln(n,' ',0,' ',0);
     halt;
  end;
  if (y div n=5000) then
  begin
    writeln(0,' ',n,' ',0);
    halt;
  end;
  if (y div n=1000) then
  begin
    writeln(0,' ',0,' ',n);
    halt;
  end;
  if (n*10000<y) then
  begin
    writeln('-1 -1 -1');
    halt;
  end;
  if (n*1000>y) then
  begin
    writeln('-1 -1 -1');
    halt;
  end;
  if (y mod 1000<>0) then
  begin
    writeln('-1 -1 -1');
    halt;
  end;
  for i:=0 to y div 10000 do
  for j:=0 to n-i do
  if (i*10000+j*5000+(n-i-j)*1000=y) 
  then begin
     writeln(i,' ',j,' ',n-i-j);
     halt;
  end;
  writeln('-1 -1 -1');
end.
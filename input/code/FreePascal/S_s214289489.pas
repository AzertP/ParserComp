{$R-,S-,Q-,I-,O+}
var n,h,left:int64;
begin
  readln(n);
  while n>0 do begin
    read(h);
    if left < h then
      left:=h-1
    else if left = h then
    else begin
      writeln('No');
      exit;
    end;
    dec(n);
  end;
  writeln('Yes');
end.
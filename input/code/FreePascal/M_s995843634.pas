var
  t,x:array[0..300005]of int64;
  n,i:longint;
  l,z,c0,c1,c2:int64;
  f,g:boolean;
begin
  readln(n,l);
  for i:=1 to n do
    read(x[i]);
  readln;
  z:=1;
  for i:=1 to n do
  begin
    read(t[i]);
    z:=z+t[i] div (2*l);
    t[i]:=t[i] mod (2*l);
    if t[i]>0 then
    begin
      inc(z);
      f:=false;
      g:=false;
      if t[i]<=(l-x[i])*2 then
        f:=true;
      if t[i]<=x[i]*2 then
        g:=true;
      if i=n then
      begin
        if f then
          dec(z);
        break;
      end;
      if g then
      begin
        inc(c0);
        if f then
          inc(c2);
      end else
      if f then 
        inc(c1);
      while c0-c2<c1+c2 do
      begin
        if c2>0 then
          dec(c2) else
          dec(c1);
      end;
    end;
  end;
  z:=z-c1-c2;
  writeln(z*2*l);
end.

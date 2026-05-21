var  a,b,c,d:longint;
     s:string;
begin
  readln(s);
  a:=ord(s[1])-ord('0');
  b:=ord(s[2])-ord('0');
  c:=ord(s[3])-ord('0');
  d:=ord(s[4])-ord('0');
  if a+b+c+d=7 then
  begin
    writeln(a,'+',b,'+',c,'+',d,'=7');
    halt;
  end;
  if a+b+c-d=7 then
  begin
    writeln(a,'+',b,'+',c,'-',d,'=7');
    halt;
  end;
  if a+b-c+d=7 then
  begin
    writeln(a,'+',b,'-',c,'+',d,'=7');
    halt;
  end;
  if a+b-c-d=7 then
  begin
    writeln(a,'+',b,'-',c,'-',d,'=7');
    halt;
  end;
  if a-b+c+d=7 then
  begin
    writeln(a,'-',b,'+',c,'+',d,'=7');
    halt;
  end;
  if a-b+c-d=7 then
  begin
    writeln(a,'-',b,'+',c,'-',d,'=7');
    halt;
  end;
  if a-b-c+d=7 then
  begin
    writeln(a,'-',b,'-',c,'+',d,'=7');
    halt;
  end;
  if a-b-c-d=7 then
  begin
    writeln(a,'-',b,'-',c,'-',d,'=7');
    halt;
  end;
end.
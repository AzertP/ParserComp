var
  a,b,c,d:longint;
  a1,b1,c1,d1:char;
begin
  readln(a1,b1,c1,d1);
  a:=ord(a1)-ord('0');
  b:=ord(b1)-ord('0');
  c:=ord(c1)-ord('0');
  d:=ord(d1)-ord('0');
  if a+b+c+d=7 then writeln(a,'+',b,'+',c,'+',d,'=7')
  else if a+b+c-d=7 then writeln(a,'+',b,'+',c,'-',d,'=7')
    else if a+b-c+d=7 then writeln(a,'+',b,'-',c,'+',d,'=7')
      else if a-b+c+d=7 then writeln(a,'-',b,'+',c,'+',d,'=7')
        else if a-b-c+d=7 then writeln(a,'-',b,'-',c,'+',d,'=7')
          else if a-b+c-d=7 then writeln(a,'-',b,'+',c,'-',d,'=7')
            else if a+b-c-d=7 then writeln(a,'+',b,'-',c,'-',d,'=7')
              else writeln(a,'-',b,'-',c,'-',d,'=7');
end.
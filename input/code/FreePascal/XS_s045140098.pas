var
a,b,c,d:longint;
begin
read(a,b,c,d);
a:=a+b;
c:=c+d;
if a=c then writeln('Balanced')
else
if a>c then writeln('Left')
else
writeln('Right');
end.
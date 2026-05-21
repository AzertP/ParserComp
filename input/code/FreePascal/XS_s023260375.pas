program C;
var x,y,w,h:longint;
    re:real;
begin
	readln(w,h,x,y);
	re:=w*h/2;
	writeln(re:0:9);
	if (x=(w/2)) and (y=(h/2)) then write(1)
	else write(0);
end.
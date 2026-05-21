var
	n,h,w,i,x,y:longint;
    ans:int64;
begin
	readln(n,h,w);
    ans:=0;
    for i:=1 to n do
    begin
    	read(x,y);
        if (x>=h) and (y>=w) then inc(ans);
    end;
    writeln(ans);
end.
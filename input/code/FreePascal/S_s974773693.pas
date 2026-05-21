var
	a,b,ans:int64;
function f(x:int64):int64;
begin
	if x mod 2=0 then 
	begin 
		if x div 2 mod 2=0 then exit(x)
		else exit(x xor 1);
	end
	else
	begin 
		if (x+1) div 2 mod 2=0 then exit(0)
		else exit(1);
	end;
end;
begin
	readln(a,b);
	ans:=f(a-1) xor f(b);
	writeln(ans);
end.
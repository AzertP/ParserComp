program ideone;
var
	l, k, n, i, q : integer;
	p : string;
begin
	readln(n,k);
	l := 1;
	for i := 1 to n do
	begin
		if (l*2) > (l+k) then
		begin
			l := l+k;
		end
		else begin
			l := l*2;
		end;
	end;
	writeln(l);
end.
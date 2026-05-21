var
	A,B,C,D : integer;
begin
	readln(A, B, C, D);
	if ((A*B) > (C*D)) then
		begin
			writeln(A*B);
		end
		else 
			begin writeln(C*D);
	end;
end.
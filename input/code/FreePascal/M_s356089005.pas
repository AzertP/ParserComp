var
	i,j,h,w,n,a,c,t:Longint;
	u:array[1..100]of Longint;
begin
	t:=1;
	read(h,w,n,a);
	for i:=1 to h do begin
		if i mod 2=1 then begin
			for j:=1 to w do begin
				writeln(t);
				inc(c);
				if a=c then begin
					read(a);
					c:=0;
					inc(t);
				end;
			end;
		end else begin
			for j:=w downto 1 do begin
				u[j]:=t;
				inc(c);
				if a=c then begin
					read(a);
					c:=0;
					inc(t);
				end;
			end;
			for j:=1 to w do writeln(u[j]);
		end;
	end;
end.

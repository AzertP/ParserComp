var
	s,t:String;
	c,i:Longint;
begin
	readln(s);
	for i:=length(s)downto 1 do begin
		if s[i]='B' then inc(c)else if c>0 then dec(c) else t:=s[i]+t;
	end;
	writeln(t);
end.
var
	c,i:Longint;
	S:String[4];
	cnt:Array[1..600]of Longint;
begin
	readln(S);
	for i:=1 to length(S) do begin
		inc(cnt[ord(S[i])]);
		if cnt[ord(S[i])]=2 then inc(c);
	end;
	if c=2 then writeln('Yes')else writeln('No');
end.

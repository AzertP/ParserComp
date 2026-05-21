var s:String;i:Longint;
begin
readln(s);
i:=1;
while i<=length(s) do begin
	write(s[i]);
	inc(i,2);
end;
end.

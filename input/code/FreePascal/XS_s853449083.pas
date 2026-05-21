var
	n,i:Longint;
	s:String;
begin
	setLength(s,100000);
	readln(s);
	n:=length(s);
	i:=1;
	while(i<n)and not((s[i]=s[i+1])or((i+1<n)and(s[i]=s[i+2])))do inc(i);
	if i=n then writeln('-1 -1')else if s[i]=s[i+1] then writeln(i,' ',i+1)else writeln(i,' ',i+2);
end.

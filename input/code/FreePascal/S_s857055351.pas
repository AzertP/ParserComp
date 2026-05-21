var
	a:array[0..100010]of longint;
	i,n,k,x,y:longint;
	s:int64;
function min(x,y:longint):longint;
begin
	if x<y then exit(x) else exit(y);
end;
begin
	readln(n,k);
	for i:=1 to n do read(a[i]);
	s:=maxlongint;
	for i:=1 to n-k+1 do
	begin
		x:=abs(a[i]);
		y:=abs(a[k+i-1]);
		if a[i]<=0 then
			if a[k+i-1]<=0 then s:=min(s,-a[i])
			else s:=min(s,min(x*2+y,y*2+x))
		else s:=min(s,a[k+i-1]);
	end;
	writeln(s);
end.
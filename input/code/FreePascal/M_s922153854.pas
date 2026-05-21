program ec12;
var 
	a,b,site:array[1..100001] of int64;
	j:int64;
	n,m,i,max,tot:longint;
procedure qsrt(l,r:longint);
var 
	s,v:int64;
	i,j:longint;
begin 
	i:=l; j:=r; v:=a[(l+r) div 2];
	repeat 
	while a[i]<v do inc(i);
	while a[j]>v do dec(j);
	if i<=j then 
	begin 
		s:=a[i];
		a[i]:=a[j];
		a[j]:=s;
		inc(i);
		dec(j);
	end;
	until i>=j;
	if l<j then qsrt(l,j);
	if i<r then qsrt(i,r);
end; 
begin 
	readln(n);
	for i:=1 to n do 
	read(a[i]);
	qsrt(1,n);
	m:=0;
	a[n+1]:=0;	
	for i:=n downto 1 do 
	begin
		if a[i]=a[i+1] then 
		begin 
			if m=0 then 
			begin 
				m:=1;
				j:=a[i];
				a[i]:=-1;
			end
			else
			begin 
				writeln(j*a[i]);
				halt;
			end;
		end;
	end;
	writeln(0);
end. 
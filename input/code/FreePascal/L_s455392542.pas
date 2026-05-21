program ec12;
var 
	n,m,i,j:longint;
	ans:int64;
	a,b,c,s,sum:array[0..100001] of int64;
procedure qsrt(l,r:longint);
var 
	v,s:int64;
	i,j:longint;
begin
	i:=l; j:=r; v:=b[(l+r) div 2];
	repeat
	while b[i]<v do inc(i);
	while b[j]>v do dec(j);
	if i<=j then 
	begin 
		s:=b[i];
		b[i]:=b[j];
		b[j]:=s;
		inc(i);
		dec(j);
	end;
	until i>=j;
	if l<j then qsrt(l,j);
	if i<r then qsrt(i,r);
end;
procedure qsrt1(l,r:longint);
var 
	v,s:int64;
	i,j:longint;
begin
	i:=l; j:=r; v:=c[(l+r) div 2];
	repeat
	while c[i]<v do inc(i);
	while c[j]>v do dec(j);
	if i<=j then 
	begin 
		s:=c[i];
		c[i]:=c[j];
		c[j]:=s;
		inc(i);
		dec(j);
	end;
	until i>=j;
	if l<j then qsrt1(l,j);
	if i<r then qsrt1(i,r);
end;
procedure qsrt2(l,r:longint);
var 
	v,s:int64;
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
	if l<j then qsrt2(l,j);
	if i<r then qsrt2(i,r);
end;
begin 
	readln(n);
	for i:=1 to n do
	read(a[i]);
	qsrt2(1,n);
	for i:=1 to n do
	read(b[i]);
	qsrt(1,n);
	for i:=1 to n do
	read(c[i]);
	qsrt1(1,n);
	i:=1;
	j:=1;
	while (i<=n) and (j<=n) do
	begin 
		if b[i]>=c[j] then 
		inc(j)
		else
		begin 
			s[i]:=n-j+1;
			inc(i);
		end;
	end;
	i:=1;
	j:=1;
	sum[0]:=0;
	for i:=1 to n do 
	sum[i]:=sum[i-1]+s[i];
	ans:=0;
	i:=1;
	j:=1;
	while (i<=n) and (j<=n) do 
	begin 
		if a[i]>=b[j] then 
		inc(j)
		else
		begin 
			ans:=ans+sum[n]-sum[j-1];
			inc(i);
		end;
	end;
	writeln(ans);
end. 
program ec12;
var 
	k:qword;
	a:array[1..101] of qword;
	n,i:longint;
	ans:qword;
procedure qsrt(l,r:longint);
var 
	v,s:qword;
	i,j:longint;
begin 
	i:=l; j:=r; v:=a[(l+r) div 2];
	repeat
	while a[i]>v do inc(i);
	while a[j]<v do dec(j);
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
function gcd(a,b:qword):qword;
var 
	c:qword;
begin 
	while true do 
	begin
		if b=0 then 
		exit(a);
		c:=a;
		a:=b;
		b:=c mod b;
	end;
end;
begin 
	readln(n);
	for i:=1 to n do 
	readln(a[i]);
	if n=1 then 
	begin 	
		writeln(a[1]);
		halt;
	end;
	qsrt(1,n);
	if a[1]=a[2] then 
	ans:=a[1]
	else
	begin 
		k:=gcd(a[1],a[2]);
		ans:=(a[1] div k)*a[2];
	end;
	for i:=3 to n do 
	begin 
		if ans mod a[i]=0 then 
		continue;
		k:=gcd(ans,a[i]);
		ans:=(ans div k)*a[i];
	end;
	writeln(ans);
end. 
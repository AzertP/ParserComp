var
	c,w,f,p:array[0..100000]of longint;
	a:array[0..1000,0..100010]of longint; 
	g,v,x,y,d,n,i,j,k,t:longint;
begin
	readln(d,g);
	v:=g div 100;
	n:=0;
	fillchar(a,sizeof(a),0);
	fillchar(f,sizeof(f),0);
	for i:=1 to d do
    begin
		readln(x,y);
		y:=y div 100;
		for j:=1 to x do
	    begin
			inc(n);
			if j<>x then w[n]:=i*j else w[n]:=i*j+y;
			c[n]:=j; p[n]:=i;
		end;
	end;
	for i:=1 to n do
    begin
		inc(a[p[i],0]);
		a[p[i],a[p[i],0]]:=i;
	end;
	for i:=1 to n do
		for j:=v downto 0 do
			for k:=1 to a[i,0] do
				if j>=c[a[i,k]] then
					if f[j]<f[j-c[a[i,k]]]+w[a[i,k]] then
						f[j]:=f[j-c[a[i,k]]]+w[a[i,k]]; //分组背包
	for i:=1 to n do
		if f[i]>=v then break;
	writeln(i);
end.
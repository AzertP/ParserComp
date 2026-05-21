var n,k,i,aa:longint;
min,count:int64;
a:array[0..100010]of longint;
function m(x,y:int64):int64;
begin 
	if x<y then exit(x)
		else exit(y);
end;
begin 
	readln(n,k);
	aa:=0;
	min:=maxlongint;
	for i:=1 to n do 
	begin
		read(a[i]);
		if a[i]<=0 then 
		begin 
			a[i]:=-a[i];aa:=i;
		end;
	end;
	if aa>=k then min:=m(min,a[aa-k+1]);
	if n-aa>=k then min:=m(min,a[aa+k]);
	for i:=0 to m(n-aa,k) do 
		if (aa>=k-i) then
		begin 
			if a[aa-k+i+1]>a[aa+i] then count:=a[aa-k+i+1]+2*a[aa+i]
				else count:=2*a[aa-k+i+1]+a[aa+i];
			min:=m(min,count);
		end;
	write(min);
end.
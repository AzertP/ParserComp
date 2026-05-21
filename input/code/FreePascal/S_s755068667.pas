uses math;
var
	ans,n,Ma,Mb,i,j,a,b,c,x,mp,mm:Longint;
	p,m:array[0..100]of Longint;
begin
	read(n,Ma,Mb);
	ans:=10000;
	for i:=1 to 100 do begin
		p[i]:=ans;
		m[i]:=ans;
	end;
	for i:=1 to n do begin
		read(a,b,c);
		x:=Ma*b-Mb*a;
		if x>0 then begin
			mp:=min(mp+x,100);
			for j:=mp-x downto 0 do p[j+x]:=min(p[j+x],p[j]+c);
		end else if x<0 then begin
			mm:=min(mm-x,100);
			for j:=mm+x downto 0 do m[j-x]:=min(m[j-x],m[j]+c);
		end else ans:=min(ans,c);
	end;
	for i:=1 to 100 do ans:=min(ans,p[i]+m[i]);
	if ans<10000 then writeln(ans)else writeln(-1);
end.
var
	N,i,j,k:Longint;
	A,B,C,S,cx,cy,cr:double;
	x,y:Array[1..50]of double;
function norm(x1,y1,x2,y2:double):double;
begin
	norm:=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
end;
function ok(id:Longint):Boolean;
begin
	ok:=norm(x[id],y[id],cx,cy)<cr+0.0000000001;
end;
procedure make(ia,ib:Longint);
begin
	cx:=(x[ia]+x[ib])/2;
	cy:=(y[ia]+y[ib])/2;
	cr:=norm(x[ia],y[ia],cx,cy);
end;
begin
	read(N);
	for i:=1 to N do read(x[i],y[i]);
	make(1,2);
	for i:=3 to N do if not ok(i)then begin
		make(1,i);
		for j:=2 to i-1 do if not ok(j)then begin
			make(i,j);
			for k:=1 to j-1 do if not ok(k)then begin
				A:=norm(x[j],y[j],x[k],y[k]);
				B:=norm(x[k],y[k],x[i],y[i]);
				C:=norm(x[i],y[i],x[j],y[j]);
				S:=(x[j]-x[i])*(y[k]-y[i])-(x[k]-x[i])*(y[j]-y[i]);
				S:=4*S*S;
				cx:=(A*(B+C-A)*x[i]+B*(C+A-B)*x[j]+C*(A+B-C)*x[k])/S;
				cy:=(A*(B+C-A)*y[i]+B*(C+A-B)*y[j]+C*(A+B-C)*y[k])/S;
				cr:=norm(x[i],y[i],cx,cy);
			end;
		end;
	end;
	writeln(sqrt(cr):0:9);
end.

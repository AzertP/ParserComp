uses Math;
var
	N,K,i,j:Longint;
	x,y,c:Array[1..60]of Longint;
	rr:Array[1..60]of double;
	L,R,M,a,t,d,ri,rj,yy:double;
	flag:Boolean;
function count(tx,ty:double):Boolean;
var i,cnt:Longint;
begin
	cnt:=0;
	for i:=1 to N do if sqr(X[i]-tx)+sqr(Y[i]-ty)<sqr(rr[i])+0.000000001 then begin
		inc(cnt);
		if cnt=K then begin
			count:=True;
			exit;
		end;
	end;
	count:=False;
end;
begin
	read(N,K);
	for i:=1 to N do read(x[i],y[i],c[i]);
	L:=0;
	R:=200000;
	while R-L>0.000001 do begin
		M:=(L+R)/2;
		for i:=1 to N do rr[i]:=M/c[i];
		flag:=False;
		for i:=1 to N do begin
			if count(X[i],Y[i])then begin
				flag:=True;
				break;
			end;
		end;
		if not flag then begin
			for i:=2 to N do begin
				for j:=1 to i-1 do begin
					d:=sqr(X[i]-X[j])+sqr(Y[i]-Y[j]);
					ri:=rr[i];
					rj:=rr[j];
					yy:=4*sqr(ri)*d-sqr(sqr(ri)+d-sqr(rj));
					if yy<0 then continue;
					a:=arctan2(sqrt(yy),sqr(ri)+d-sqr(rj));
					t:=arctan2(Y[j]-Y[i],X[j]-X[i]);
					if count(X[i]+ri*cos(t+a),Y[i]+ri*sin(t+a)) then flag:=True
					else if count(X[i]+ri*cos(t-a),Y[i]+ri*sin(t-a))then flag:=True;
					if flag then break;
				end;
				if flag then break;
			end;
		end;
		if flag then R:=M else L:=M;
	end;
	writeln(R:0:9);
end.

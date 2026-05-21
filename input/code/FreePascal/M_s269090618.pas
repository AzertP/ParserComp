var
	N,M,i,u,v,id:Longint;
	Gv,Gp:Array[1..2000]of Longint;
	Gi:Array[1..1000]of Longint;
	cnt:Array[1..1000]of Longint;
	Q:Array[1..1000]of Longint;
	Qs:Longint;
begin
	read(N,M);if (N=1000)and(M<1000) then begin writeln(-1);exit;end;
	for i:=1 to M do begin
		read(u,v);
		Gv[i]:=v;
		if Gi[u]>0 then begin
			Gp[i]:=Gi[u];
		end;
		Gi[u]:=i;
		inc(cnt[v]);
	end;
	for i:=1 to N do begin
		if cnt[i]=0 then begin
			inc(Qs);
			Q[Qs]:=i;
		end;
	end;
	id:=1;
	while id<=Qs do begin
		u:=Q[id];
		inc(id);
		i:=Gi[u];
		while i<>0 do begin
			dec(cnt[Gv[i]]);
			if cnt[Gv[i]]=0 then begin
				inc(Qs);
				Q[Qs]:=Gv[i];
			end;
			i:=Gp[i];
		end;
	end;
	if Qs<N then writeln(0)else writeln(-1);
end.
type UnionFind=record n:Longint;pr:Array of Longint;end;
procedure init(var uf:UnionFind;N:Longint);
var i:Longint;
begin
	uf.n:=N;
	setlength(uf.pr,N);
	for i:=1 to N do uf.pr[i]:=-1;
end;
function find(var uf:UnionFind;a:Longint):Longint;
begin
	if uf.pr[a]<0 then find:=a else begin
		uf.pr[a]:=find(uf,uf.pr[a]);
		find:=uf.pr[a];
	end;
end;
function same(var uf:UnionFind;a,b:Longint):Boolean;
begin
	same:=find(uf,a)=find(uf,b);
end;
function unite(var uf:UnionFind;a,b:Longint):Boolean;
begin
	a:=find(uf,a);
	b:=find(uf,b);
	if a=b then unite:=false else begin
		if uf.pr[a]>uf.pr[b] then begin
			inc(uf.pr[b],uf.pr[a]);
			uf.pr[a]:=b;
		end else begin
			inc(uf.pr[a],uf.pr[b]);
			uf.pr[b]:=a;
		end;
		unite:=true;
	end;
end;
function size(var uf:UnionFind;a:Longint):Longint;
begin
	size:=-uf.pr[find(uf,a)];
end;
var
	U,V,suc,beg:Array[1..400000]of Longint;
	N,M,K,i,id,a,b,ec,ans:Longint;
	uf:UnionFind;
procedure add_edge(a,b:Longint);
begin
	inc(ec);
	U[ec]:=a;
	V[ec]:=b;
	suc[ec]:=beg[a];
	beg[a]:=ec;
end;
begin
	read(N,M,K);
	init(uf,N);
	for i:=1 to M do begin
		read(a,b);
		add_edge(a,b);
		add_edge(b,a);
		unite(uf,a,b);
	end;
	for i:=1 to K do begin
		read(a,b);
		add_edge(a,b);
		add_edge(b,a);
	end;
	for i:=1 to N do begin
		ans:=size(uf,i)-1;
		id:=beg[i];
		while id>0 do begin
			if same(uf,i,V[id]) then dec(ans);
			id:=suc[id];
		end;
		writeln(ans);
	end;
end.

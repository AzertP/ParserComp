type X=record a,b:int64;f:Boolean;end;
type SortDataType=X;
function SortCmp(const a,b:SortDataType):Boolean;forward;
procedure SortImpl(var A:Array of SortDataType;L,R:Longint);
var
	tmp:SortDataType;
	mid,il,ir:Longint;
begin
	if R-L<20 then begin
		for ir:=R-1 downto L+1 do for il:=L to ir-1 do if SortCmp(A[il+1],A[il])then begin
			tmp:=A[il];
			A[il]:=A[il+1];
			A[il+1]:=tmp;
		end;
	end else begin
		mid:=(L+R)div 2;
		if SortCmp(A[mid],A[L])then begin
			tmp:=A[mid];
			A[mid]:=A[L];
			A[L]:=tmp;
		end;
		if SortCmp(A[R-1],A[mid])then begin
			tmp:=A[R-1];
			A[R-1]:=A[mid];
			A[mid]:=tmp;
			if SortCmp(A[mid],A[L])then begin
				tmp:=A[mid];
				A[mid]:=A[L];
				A[L]:=tmp;
			end;
		end;
		il:=L;
		ir:=R-1;
		while il<=ir do begin
			while SortCmp(A[il],A[mid])do inc(il);
			while SortCmp(A[mid],A[ir])do dec(ir);
			if il<ir then begin
				tmp:=A[il];
				A[il]:=A[ir];
				A[ir]:=tmp;
				if mid=il then mid:=ir else if mid=ir then mid:=il;
				inc(il);
				dec(ir);
			end else if il=ir then begin
				inc(il);
				dec(ir);
			end;
		end;
		SortImpl(A,L,il);
		SortImpl(A,il,R);
	end;
end;
procedure Sort(var A:Array of SortDataType;size:Longint);
begin
	SortImpl(A,0,size);
end;
function SortCmp(const a,b:SortDataType):Boolean;
begin
	if a.a<>b.a then SortCmp:=a.a<b.a else if a.b<>b.b then SortCmp:=a.b<b.b else SortCmp:=False;
end;
const M=1000000007;
var
	N,i,j,sz:Longint;
	f:Boolean;
	a,b,g,tmp,u,v,ans,O:int64;
	Ar:Array[1..200000]of X;
function gcd(a,b:int64):int64;
begin
	if b=0 then gcd:=a else gcd:=gcd(b,a mod b);
end;
begin
	read(N);
	O:=M-1;
	sz:=0;
	for i:=1 to N do begin
		read(a,b);
		if(a=0)and(b=0)then inc(O)else begin
			g:=gcd(a,b);
			a:=a div g;
			b:=b div g;
			if(a<0)or((a=0)and(b>0))then begin
				a:=-a;
				b:=-b;
			end;
			f:=False;
			if b<0 then begin
				tmp:=a;
				a:=-b;
				b:=tmp;
				f:=True;
			end;
			inc(sz);
			Ar[sz].a:=a;
			Ar[sz].b:=b;
			Ar[sz].f:=f;
		end;
	end;
	Sort(Ar,sz);
	i:=1;
	ans:=1;
	while i<=sz do begin
		j:=i;
		u:=1;
		v:=1;
		while(i<=sz)and(Ar[i].a=Ar[j].a)and(Ar[i].b=Ar[j].b)do begin
			if Ar[i].f then u:=u*2 mod M else v:=v*2 mod M;
			inc(i);
		end;
		ans:=ans*(u+v-1)mod M;
	end;
	writeln((ans+O)mod M);
end.

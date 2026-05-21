type SortDataType=String;
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
	SortCmp:=a<b;
end;
var
	N,i,cnt,ac,id:Longint;
	S:Array[1..200000]of String;
	ast:Array[1..200000]of String;
begin
	readln(N);
	for i:=1 to N do readln(S[i]);
    Sort(S,N);
	cnt:=0;
	id:=0;
	for i:=1 to N do begin
		if(i>1)and(S[i-1]<>S[i])then cnt:=0;
		inc(cnt);
		if ac<cnt then begin
			ac:=cnt;
			id:=1;
			ast[id]:=S[i];
		end else if ac=cnt then begin
			inc(id);
			ast[id]:=S[i];
		end;
	end;
	for i:=1 to id do writeln(ast[i]);
end.

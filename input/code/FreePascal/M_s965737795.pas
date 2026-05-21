var
	Q,i,j:Longint;
	T:Array[0..30]of int64;
	a,b,c,d,aa,bb,cc,dd,ans,plus,tmp:int64;
begin
	T[0]:=1;
	for i:=1 to 30 do T[i]:=T[i-1]*3;
	read(Q);
	for i:=1 to Q do begin
		read(a,b,c,d);
		dec(a);
		dec(b);
		dec(c);
		dec(d);
		ans:=abs(a-c)+abs(b-d);
		plus:=0;
		for j:=30 downto 1 do begin
			if(a div T[j-1]=c div T[j-1])and(a div T[j-1]mod 3=1)then begin
				bb:=b div T[j-1];
				dd:=d div T[j-1];
				if bb>dd then begin
					tmp:=bb;
					bb:=dd;
					dd:=tmp;
				end;
				while bb mod 3<>1 do inc(bb);
				if bb<=dd then begin
					a:=a mod T[j];
					c:=c mod T[j];
					if a>c then begin
						tmp:=a;
						a:=c;
						c:=tmp;
					end;
					if a+c+1<3*T[j-1]then plus:=a+1-T[j-1]else plus:=2*T[j-1]-c;
					break;
				end;
			end;
			if(b div T[j-1]=d div T[j-1])and(b div T[j-1]mod 3=1)then begin
				aa:=a div T[j-1];
				cc:=c div T[j-1];
				if aa>cc then begin
					tmp:=aa;
					aa:=cc;
					cc:=tmp;
				end;
				while aa mod 3<>1 do inc(aa);
				if aa<=cc then begin
					b:=b mod T[j];
					d:=d mod T[j];
					if b>d then begin
						tmp:=b;
						b:=d;
						d:=tmp;
					end;
					if b+d+1<3*T[j-1]then plus:=b+1-T[j-1]else plus:=2*T[j-1]-d;
					break;
				end;
			end;
		end;
		writeln(ans+plus*2);
	end;
end.

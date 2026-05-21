var
	i,j,k,n:Longint;
	a:array[1..50]of int64;
	L,ans:int64;
	flag:Boolean;
begin
	read(n);
	for i:=1 to n do read(a[i]);
	while true do begin
		for i:=n downto 2 do begin
			flag:=false;
			for j:=1 to i-1 do begin
				if a[j]<a[j+1]then begin
					L:=a[j];
					a[j]:=a[j+1];
					a[j+1]:=L;
					flag:=true;
				end;
			end;
			if not flag then break;
		end;
		if a[1]<n then break;
		k:=1;
		while(k+1<=n)and(a[k+1]+k>=n)and(a[k]-a[k+1]<=n)do inc(k);
		if k=n then begin
			k:=1;
			while(k+1<=n)and(a[1]-a[k+1]<=k)do inc(k);
		end;
		if k<n then begin
			L:=(a[1]-a[k+1])div(n+1);
			if L<=0 then L:=1;
			inc(ans,L*k);
			for i:=1 to n do begin
				if i>k then inc(a[i],L*k)
				else inc(a[i],L*(k-1-n));
			end;
		end else begin
			L:=a[1]-n+1;
			inc(ans,L*n);
			writeln(ans);
			exit;
		end;
	end;
	writeln(ans);
end.
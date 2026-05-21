var
	N,i,j,ans:int64;
begin
	read(N);
	i:=1;
	while i<=N do begin
		j:=N div(N div i)+1;
		inc(ans,(j-i)*(i+j-1)*(N div i)*(N div i+1)div 4);
		i:=j;
	end;
	writeln(ans);
end.
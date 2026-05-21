program PopularVote;
var
	N, M, i, count : integer;
	total : int64;
	r :real;
	a : array[1..100] of integer;
begin
	readln(N, M); count := 0; total := 0;
	
	for i := 1 to N do begin
		read(a[i]);
		total := total + a[i] ;
	end;
	
	r := total*(1 / (4 * M));
	for i := 1 to N do 
		if a[i] >= r 
			then count := count + 1;
			
	if count >= M 
		then writeln('Yes')
		else writeln('No');
end.

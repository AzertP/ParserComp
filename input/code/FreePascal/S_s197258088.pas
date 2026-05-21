program Lucky7;
var
	n : integer;
	contain : boolean;
begin
	readln(n);
	if ( n div 100 = 7 )
		then contain := true
		else if ( ((n div 10) mod 10) = 7 )
				then contain := true 
				else if ( (n mod 10) = 7 )
						then contain := true 
						else contain := false;
	if contain
		then writeln('Yes')
		else writeln('No');
end.
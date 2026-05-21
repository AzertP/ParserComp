program main;
var
m, n, i, total, count : integer;
a : array[1..1000] of integer;

begin
	read(n, m);
    total := 0;
    for i := 1 to n do begin
    	read(a[i]);
        total := total + a[i];
    end;
    
    
    count := 0;
    
    for i := 1 to n do begin
        if (a[i]/total) >= (1/(4*m))
            then count := count + 1
    end;
    if count >= m 
        then writeln('Yes')
            else writeln('No');
end.
program main;
var 
   a : integer;
begin
    readln(a);
    if a < 1000 then 
        writeln(1000-a)
    else 
        begin 
            while ( a > 1000 ) do
                dec(a,1000);
            if a = 0 then 
                writeln('0')
            else
                writeln(1000-a);
        end;
end.
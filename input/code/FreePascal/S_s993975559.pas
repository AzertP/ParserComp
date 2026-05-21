program main;
var 
    a,i,b : integer;
    p : boolean;
begin 
    p := true;
    readln(a);
    for i := 1 to a do 
        begin 
            read(b);
            if ( b mod 2 ) = 0 then 
                begin 
                    if (( b mod 3 ) <> 0) then
                        if (( b mod 5 ) <> 0) then
                            p := false;
                end;
        end;
    if ( p = true ) then 
        writeln('APPROVED')
    else 
        writeln('DENIED');
end.
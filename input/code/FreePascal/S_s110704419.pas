program trickortreat;
var
    snuke : array[1..100] of boolean;
    n, k, i, j, x, y, counter : longint;
begin
    readln(n, k);
    counter := 0;
    for i := 1 to 100 do begin
        snuke[i] := false;
    end;
    
    for i := 1 to k do begin
        readln(x);
        for j := 1 to x do begin
            read(y);
            snuke[y] := true;
        end;
    end;    

    for i := 1 to n do begin
        if not(snuke[i]) then       
            counter += 1;
    end;
    writeln(counter);
end.
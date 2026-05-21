var s, s1, s2 : ansistring;
    n : longint;
    
function check(s : ansistring) : boolean;
var n, i, j : longint;
begin
    n := length(s);
    i := 1; j := n;
    while (i < j) do
    begin
        if (s[i] <> s[j]) then exit(false);
        inc(i); dec(j);
    end;
    exit(true);
end;

begin
    readln(s);
    n := length(s);
    s1 := copy(s, 1, (n - 1) div 2);
    s2 := copy(s, (n + 3) div 2, (n - 1) div 2);
    if (check(s) and check(s1) and check(s2)) then writeln('Yes')
    else writeln('No');
end.
var
    i,l,n,r:longint;
    ans:longint;
begin
    readln(n);
    for i:=1 to n do
        begin
            readln(l,r);
            ans:=ans+r-l;
        end;
    writeln(ans+n);
end.
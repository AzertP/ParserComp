var
        i,k:longint;
        n:ansistring;
begin
        readln(n);
        for i:=1 to length(n) do
            if k=0 then begin if n[i]='C' then inc(k); end
            else if k=1 then begin if n[i]='F' then inc(k); end
            else break;
        if k=2 then writeln('Yes')
        else writeln('No');
end.
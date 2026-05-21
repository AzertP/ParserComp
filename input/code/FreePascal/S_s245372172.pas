var
    l,i,t,k,cnt,n:longint;
begin
    readln(l);n:=l;
    while l>0 do
        begin
            inc(k);
            if l mod 2=1 then inc(cnt);
            l:=l>>1;
        end;
    dec(cnt);
    writeln(k,' ',2*(k-1)+cnt);
    t:=1;
    for i:=1 to k-1 do
        begin
            writeln(i,' ',i+1,' ',0);
            writeln(i,' ',i+1,' ',t);
            t:=t*2;
        end;
    cnt:=k-1;dec(n,1<<cnt);
    for i:=k-2 downto 0 do
        begin 
            if n>=1<<i then 
                begin
                    dec(n,1<<i);
                    writeln(cnt,' ',k,' ',t);
                    inc(t,1<<i);
                end;
            dec(cnt);
        end;
end.
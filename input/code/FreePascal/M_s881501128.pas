var
    s:string;
    i,cnt:longint;
begin
    readln(s);
    if s[1]<>'A' then 
        begin
            writeln('WA');
            exit;
        end;
    if not((s[2]>='a') and (s[2]<='z') and (s[length(s)]>='a') and (s[length(s)-1]<='z')) then 
        begin
            writeln('WA');
            exit;
        end;
    for i:=3 to length(s)-1 do
        if s[i]='C' then
            begin
                inc(cnt);
                if cnt>=2 then 
                    begin
                        writeln('WA');
                        exit;
                    end;
            end
        else if not((s[i]>='a') and (s[i]<='z')) then 
            begin
                writeln('WA');
                exit;
            end;
    if cnt=1 then writeln('AC')
    else writeln('WA');
end.
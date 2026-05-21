var
    a,b,len,i: Integer;
    s:String;

begin
    readln(a,b);
    len:=a+b+1;
    readln(s);
    if s[a+1]<>'-' then
    begin
        writeln('No');
        readln;
        exit;
    end;
    for i := 1 to a do
    begin
        if not ((s[i]>='0') and (s[i]<='9')) then
        begin
            writeln('No');
            readln;
            exit;
        end;
    end; 
    for i := a+2 to len do 
    begin
        if not ((s[i]>='0') and (s[i]<='9')) then
        begin
            writeln('No');
            readln;
            exit;
        end;
    end;
    writeln('Yes');
    readln;
end.
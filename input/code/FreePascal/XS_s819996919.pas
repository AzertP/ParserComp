var i:longint;
    s:string;
    a,b:boolean;
begin
    a:=false;
    b:=false;
    readln(s);
    for i:=1 to length(s) do
    begin
        if s[i]='C' then a:=true;
        if (s[i]='F')and(a) then b:=true;
    end;
    if b then write('Yes') else write('No');
end.
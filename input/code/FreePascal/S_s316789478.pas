Var s,x:ansistring;
    n,i:longint;
Begin
        readln(s);
        for i:=1 to length(s) do
        Begin
                if s[i] in ['C','F'] then
                Begin
                        x:=x+s[i];
                end;
                
                if pos('CF',x)<>0 then
                Begin
                        write('Yes');
                        halt;
                end;
        end;
        write('No');
end.
var
    n,i,j,m,k,d,t,t0,bit_count,bit_need,s,first_ava: Integer;
    flag,flag2,flag3:Boolean;
    bit:array[1..4] of Integer;
    bit_avaliable:array[0..9] of Boolean;
    match:array[1..6] of Integer;
    match2:array[1..6] of Integer;
    bit_out:array[1..5] of Integer;

procedure my_output(n:Integer);
var
    i: Integer;
begin
    for i := 1 to n do
    begin
        write(bit_out[i]);
    end;
    writeln;
    readln;
    readln;
end;

begin
    readln(n,k);
    for i := 0 to 9 do
    begin
        bit_avaliable[i]:=true;
    end;
    for i := 1 to k do
    begin
        read(d);
        bit_avaliable[d]:=false;
    end;

    for i := 1 to 9 do
    begin//get first avaliable
        if bit_avaliable[i] then
        begin
            first_ava:=i;
            break;
        end;
    end;

    bit_count:=0;
    while n>0 do
    begin
        inc(bit_count);
        bit[bit_count]:=n mod 10;
        n:=n div 10;
    end;
    bit_need:=bit_count;

    t:=0;
    flag3:=true;
    for j := bit_count downto 1 do
    begin
        flag2:=true;
        for i := bit[j] to 9 do
        begin
            if bit_avaliable[i] then
            begin
                if i>bit[j] then
                begin
                    inc(t);
                    match[t]:=i;
                    flag3:=false;
                    break;
                end;
                flag2:=false;
                inc(t);
                match[t]:=i;
                flag:=true;
                for m := i+1 to 9 do
                begin
                    if bit_avaliable[m] then
                    begin
                        match2[t]:=m;
                        flag:=false;
                        break;
                    end;
                end;
                if flag then
                begin
                    match2[t]:=-1;
                end;
                break;
            end;
        end;
        if not flag3 then
        begin
            for s := 1 to t do
            begin
                bit_out[s]:=match[s];
            end;
            if bit_avaliable[0] then
            begin
                first_ava:=0;
            end;
            for s := t+1 to bit_need do
            begin
                bit_out[s]:=first_ava;
            end;
            my_output(bit_need);
            exit;
        end;
        if flag2 then
        begin
            t0:=t;
            while (t0>0) and (match2[t0]=-1) do
            begin
                dec(t0);
            end;
            if t0=0 then
            begin
                inc(bit_need);
                bit_out[1]:=first_ava;
                if bit_avaliable[0] then
                begin
                    first_ava:=0;
                end;
                for s := 2 to bit_need do
                begin
                    bit_out[s]:=first_ava;
                end;
                my_output(bit_need);
                exit;
            end
            else
            begin
                for s := 1 to t0-1 do
                begin
                    bit_out[s]:=match[s];
                end;
                bit_out[t0]:=match2[t0];
                if bit_avaliable[0] then
                begin
                    first_ava:=0;
                end;
                for s := t0+1 to bit_need do
                begin
                    bit_out[s]:=first_ava;
                end;
                my_output(bit_need);
                exit;
            end;
        end;
    end;

    for s := 1 to bit_need do
    begin
        bit_out[s]:=match[s];
    end;
    my_output(bit_need);
end.
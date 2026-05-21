var
    n,i,j,m,k,d,t,t0,bit_count,bit_need,s,first_ava,bit_avaliable_cnt,
    first_ava_0: Integer;
    flag:Boolean;
    bit:array[1..4] of Integer;
    bit_avaliable:array[0..9] of Boolean;
    bit_next:array[0..9,1..2] of Integer;
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
    bit_avaliable_cnt:=10;
    for i := 1 to k do
    begin
        read(d);
        bit_avaliable[d]:=false;
        dec(bit_avaliable_cnt);
    end;

    for i := 1 to 9 do
    begin//get first avaliable
        if bit_avaliable[i] then
        begin
            first_ava:=i;
            break;
        end;
    end;

    if bit_avaliable[0] then
    begin
        first_ava_0:=0;
    end
    else begin
        first_ava_0:=first_ava;
    end;

    t:=-1;
    flag:=false;
    for i := 0 to 9 do
    begin
        if bit_avaliable[i] then
        begin
            for j := t+1 to i do
            begin
                bit_next[j][1]:=i;
            end;
            if flag then //not first time
            begin
               for j := t0 to t do
               begin
                   bit_next[j][2]:=i;
               end;
            end;
            t0:=t+1;
            t:=i;
            flag:=true;
        end;
    end;

    for j := t0 to t do
    begin
        bit_next[j][2]:=-1;
    end;
    for j := t+1 to 9 do
    begin
        bit_next[j][1]:=-1;
        bit_next[j][2]:=-1;
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
    for j := bit_count downto 1 do
    begin
        if bit_next[bit[j]][1]>bit[j] then
        begin
            inc(t);
            match[t]:=bit_next[bit[j]][1];
            for s := 1 to t do
            begin
                bit_out[s]:=match[s];
            end;
            for s := t+1 to bit_need do
            begin
                bit_out[s]:=first_ava_0;
            end;
            my_output(bit_need);
            exit;
        end else if bit_next[bit[j]][1]=bit[j] then
        begin
            inc(t);
            match[t]:=bit_next[bit[j]][1];
            match2[t]:=bit_next[bit[j]][2];
        end else
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
                for s := 2 to bit_need do
                begin
                    bit_out[s]:=first_ava_0;
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
                for s := t0+1 to bit_need do
                begin
                    bit_out[s]:=first_ava_0;
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
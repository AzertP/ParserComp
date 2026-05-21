Program Rating;
uses    math;
var a,d:array[0..5000] of longint;
    n,minc,maxc,i:longint;

begin
    read(n);
    for i:=1 to n do
        read(a[i]);
    for i:=1 to n do
        case a[i] of
            1..399:inc(d[1]);
            400..799:inc(d[2]);
            800..1199:inc(d[3]);
            1200..1599:inc(d[4]);
            1600..1999:inc(d[5]);
            2000..2399:inc(d[6]);
            2400..2799:inc(d[7]);
            2800..3199:inc(d[8]);
        else inc(d[9]);
        end;
    for i:=1 to 8 do
        if d[i]<>0 then
        begin
        inc(minc)
        //maxc:=max(maxc,d[i]);
        end;
    maxc:=minc+d[9];
    if minc=0 then minc:=1;
    write(minc,' ',maxc);
end.

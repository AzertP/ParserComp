
var
    n,i,x,y: Integer;
    a: array[0..100] of Integer;

procedure qsort(l,r:Longint;data:PInteger);
var
    i,j,mid,t: Integer;
begin
    if l>=r then
    begin
        exit;
    end;
    i:=l;
    j:=r;
    mid:=data[(l+r) div 2];
    repeat
        while data[i]<mid do
        begin
            inc(i);
        end;
        while data[j]>mid do
        begin
            dec(j);
        end;
        if i<=j then
        begin
            t:=data[i];
            data[i]:=data[j];
            data[j]:=t;
            inc(i);
            dec(j);
        end;
    until i>j;
    qsort(l,j,data);
    qsort(i,r,data);
end;
begin
    readln(n);
    for i := 0 to n-1 do
    begin
        read(a[i]);
    end;
    qsort(0,n-1,@a[0]);
    x:=0;
    y:=0;
    i:=n-1;
    while i>=0 do
    begin
        inc(x,a[i]);
        dec(i,2);
    end;
    i:=n-2;
    while i>=0 do
    begin
        inc(y,a[i]);
        dec(i,2);
    end;
    writeln(x-y);
    readln;
    readln;
end.
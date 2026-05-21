var
    n,i,x,y: Integer;
    a: array[0..108] of Integer;
    
procedure qsort(l,r:Longint;data:PInteger);
//array index start at 0
var
    i,j,mid: Integer;
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
            data[0]:=data[i];
            data[i]:=data[j];
            data[j]:=data[0];
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
        read(a[i+8]);
    end;
    qsort(8,n+7,@a);
    x:=0;
    y:=0;
    i:=n+7;
    while i>=8 do
    begin
        inc(x,a[i]);
        dec(i,2);
    end;
    i:=n+6;
    while i>=8 do
    begin
        inc(y,a[i]);
        dec(i,2);
    end;
    writeln(x-y);
end.
var     n,i,g,b,l,c,d,y,o,u,r: longint;
        a:array[0..100000] of longint;
procedure xuli;
begin
        g:=0; b:=0; l:=0; c:=0; d:=0; y:=0; o:=0; r:=0; u:=0;
        for i:=1 to n do
        case a[i] of
        1..399: g:=1;
        400..799: b:=1;
        800..1199: l:=1;
        1200..1599: c:=1;
        1600..1999: d:=1;
        2000..2399: y:=1;
        2400..2799: o:=1;
        2800..3199: r:=1;
        3200..4800: inc(u);
end;
end;
procedure xuat;
begin
        if (u=g+b+l+c+d+y+o+r+u) then writeln(1,' ',u)
        else writeln(g+b+l+c+d+y+o+r,' ',g+b+l+c+d+y+o+r+u);
end;
begin
       
        readln(N);
        for i:=1 to n do read(A[i]);
        xuli;
        xuat;
      
end.

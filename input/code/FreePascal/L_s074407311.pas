program E;
    type point=record
        x:double;
        y:double;
    end;
    type circle=record
        center:point;
        rad:double;
    end;
    function distance(_a,_b:point):double; overload;
    begin
        distance:=sqrt((_a.x-_b.x)*(_a.x-_b.x)+(_a.y-_b.y)*(_a.y-_b.y));
    end;
    function distance(_a:point;_r:circle):double; overload;
        var _:double;
    begin
        _:=distance(_a,_r.center)-_r.rad;
        if _>0 then distance:=_ else distance:=0;
    end;
    function distance(_a,_b:circle):double; overload;
        var _:double;
    begin
        _:=distance(_a.center,_b.center)-_a.rad-_b.rad;
        if _>0 then distance:=_ else distance:=0;
    end;
    function min(_a,_b:double):double;
    begin
        if _a<_b then min:=_a else min:=_b;
    end;
    var S,T:point;
    var N,i,j,pt:integer;
    var C:array [1..1200] of circle;
    var DIJK:array [1..1200] of double;
    var USED:array [1..1200] of boolean;
    var map:array [0..1200,0..1200] of double;
    var M:double;
begin
    readln(S.x,S.y,T.x,T.y);
    readln(N);
    for i:=1 to N do readln(C[i].center.x,C[i].center.y,C[i].rad);
    map[0,N+1]:=distance(S,T);
    map[N+1,0]:=distance(S,T);
    for i:=1 to N do
    begin
        map[0,i]:=distance(S,C[i]);map[i,0]:=map[0,i];
        map[N+1,i]:=distance(T,C[i]);map[i,N+1]:=map[N+1,i];
    end;
    for i:=1 to N do
    begin
        map[i,i]:=0;
        for j:=i+1 to N do
        begin
            map[i,j]:=distance(C[i],C[j]);
            map[j,i]:=map[i,j];
        end;
    end;
    for i:=1 to N+1 do
    begin
        DIJK[i]:=map[0,i]; 
        used[i]:=false;       
    end;
    for j:=1 to N+1 do
    begin
        M:=1000000000000;
        for i:=1 to N+1 do
            if used[i]=false then
                if DIJK[i]<M then
                begin
                    pt:=i;
                    M:=DIJK[i];
                end;
        used[pt]:=true;
        for i:=1 to N+1 do
            if used[i]=false then
                DIJK[i]:=min(DIJK[i],DIJK[pt]+map[i,pt]);
    end;
    writeln(DIJK[N+1]:10:10);
end.
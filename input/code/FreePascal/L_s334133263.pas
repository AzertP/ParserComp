const   fi      ='';
        fo      ='';

var     n,top:longint;
        f,g:text;
        a,q,vt:array[0..300005] of longint;
        l,r:array[0..300005] of int64;

procedure       nhap;
var     i:longint;
begin
        assign(f,fi);reset(f);
        readln(f,n);
        for i:=1 to 3*n do
                read(f,a[i]);
        close(f);
end;

procedure       up(k    :longint);
var     v       :longint;
begin
        v:=q[k];
        while (a[v]<a[q[k div 2]]) do
                begin
                        q[k]:=q[k div 2];
                        vt[q[k]]:=k;
                        k:=k div 2;
                end;
        q[k]:=v;
        vt[q[k]]:=k;
end;

procedure       down(k  :longint);
var     l1,v    :longint;
begin
        v:=q[k];
        while 2*k<=top do
                begin
                        l1:=2*k;
                        if (l1<top) and (a[q[l1]]>a[q[l1+1]]) then inc(l1);
                        if a[q[l1]]>=a[v] then break;
                        q[k]:=q[l1];
                        vt[q[k]]:=k;
                        k:=l1;
                end;
        q[k]:=v;
        vt[v]:=k;
end;

procedure       put(u   :longint);
begin
        inc(top); q[top]:=u; vt[u]:=top;
        up(top);
end;

function        get     :longint;
begin
        get:=q[1];
        q[1]:=q[top]; vt[q[1]]:=1;
        dec(top);
        down(1);
end;

procedure       up1(k    :longint);
var     v       :longint;
begin
        v:=q[k];
        while (a[v]>a[q[k div 2]]) do
                begin
                        q[k]:=q[k div 2];
                        vt[q[k]]:=k;
                        k:=k div 2;
                end;
        q[k]:=v;
        vt[q[k]]:=k;
end;

procedure       down1(k  :longint);
var     l1,v    :longint;
begin
        v:=q[k];
        while 2*k<=top do
                begin
                        l1:=2*k;
                        if (l1<top) and (a[q[l1]]<a[q[l1+1]]) then inc(l1);
                        if a[q[l1]]<=a[v] then break;
                        q[k]:=q[l1];
                        vt[q[k]]:=k;
                        k:=l1;
                end;
        q[k]:=v;
        vt[v]:=k;
end;

procedure       put1(u   :longint);
begin
        inc(top); q[top]:=u; vt[u]:=top;
        up1(top);
end;

function        get1     :longint;
begin
        get1:=q[1];
        q[1]:=q[top]; vt[q[1]]:=1;
        dec(top);
        down1(1);
end;




procedure       xl;
var     tong,res:int64;
        i,u:longint;
begin
        assign(g,fo);rewrite(g);
        tong:=0;  top:=0;q[0]:=0;a[0]:=-round(1e9);
        for i:=1 to n do
        begin
                tong:=tong+a[i];
                put(i);
        end;
        l[n]:=tong;
        for i:=n+1 to 2*n do
        begin
                if (a[i]>a[q[1]]) and (top>0) then
                begin
                        u:=get;
                        l[i]:=l[i-1]+a[i]-a[u];
                        put(i);
                end else begin l[i]:=l[i-1];end;
        end;
        tong:=0;top:=0;q[0]:=0;a[0]:=round(1e9);
        for i:=3*n downto 2*n+1 do
        begin
                tong:=tong+a[i];
                put1(i);
        end;
        r[2*n+1]:=tong;
        for i:=2*n downto n+1 do
        begin
                if (a[i]<a[q[1]]) and (top>0) then
                begin
                        u:=get1;
                        r[i]:=r[i+1]+a[i]-a[u];
                        put1(i);
                end else begin r[i]:=r[i+1];end;
        end;
        res:=-round(1e18);
        for i:=n to 2*n do
                if res<l[i]-r[i+1] then res:=l[i]-r[i+1];
        write(g,res);
        close(g);
end;
begin
        nhap;
        xl;
end.





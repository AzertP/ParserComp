const   fi      ='';
        fo      ='';

var     f,g     :text;
        n,top   :longint;
        b,a,q,vt:array[0..300005] of longint;
        l1,l2   :array[0..300005] of int64;
        kq      :int64;
procedure       nhap;
var     i       :longint;
begin
        assign(f,fi); reset(f);
        readln(f,n);
        for i:=1 to 3*n do read(f,b[i]);
        close(f);
end;

{procedure       swap(i,j        :longint);
var     tg      :longint;
begin
        tg:=a[i]; a[i]:=a[j]; a[j]:=tg;
        tg:=vt[i]; vt[i]:=vt[j]; vt[j]:=tg;
end;

procedure       qs(d,c        :longint);
var     l,r     :longint;
        x       :longint;
begin
        x:=a[(d+c) div 2];
        l:=d; r:=c;
        repeat
                while a[l]<x do inc(l);
                while a[r]>x do dec(r);
                if l<=r then
                        begin
                                swap(l,r);
                                inc(l); dec(r);
                        end;
        until l>r;
        if l<c then qs(l,c);
        if d<r then qs(d,r);
end;       }

{procedure       traubo;
var     kq,tong :int64;
        i,j     :longint;
begin
        kq:=0;
        for i:=n to 2*n do
                begin
                        for j:=1 to 3*n do a[j]:=b[j];
                        qs(1,i);
                        qs(i+1,3*n);
                        tong:=0;
                        for j:=1 to n do tong:=tong+a[j];
                        for j:=3*n downto 2*n+1 do tong:=tong-a[j];
                        if tong>kq then kq:=tong;
                end;
        assign(g,fo); rewrite(g);
        writeln(g,kq);
        close(g);
end;  }

procedure       up(k    :longint);
var     v       :longint;
begin
        v:=q[k];
        while (b[v]<b[q[k div 2]]) do
                begin
                        q[k]:=q[k div 2];
                        vt[q[k]]:=k;
                        k:=k div 2;
                end;
        q[k]:=v;
        vt[q[k]]:=k;
end;

procedure       down(k  :longint);
var     l,v    :longint;
begin
        v:=q[k];
        while 2*k<=top do
                begin
                        l:=2*k;
                        if (l<top) and (b[q[l]]>b[q[l+1]]) then inc(l);
                        if b[q[l]]>=b[v] then break;
                        q[k]:=q[l];
                        vt[q[k]]:=k;
                        k:=l;
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
        while (b[v]>b[q[k div 2]]) do
                begin
                        q[k]:=q[k div 2];
                        vt[q[k]]:=k;
                        k:=k div 2;
                end;
        q[k]:=v;
        vt[q[k]]:=k;
end;

procedure       down1(k  :longint);
var     l,v    :longint;
begin
        v:=q[k];
        while 2*k<=top do
                begin
                        l:=2*k;
                        if (l<top) and (b[q[l]]<b[q[l+1]]) then inc(l);
                        if b[q[l]]<=b[v] then break;
                        q[k]:=q[l];
                        vt[q[k]]:=k;
                        k:=l;
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

procedure       xuli;
var     i,j     :longint;
        tong    :int64;
begin
        b[0]:=-1; q[0]:=0; top:=0; tong:=0;
        for i:=1 to n do
                begin
                        tong:=tong+b[i];
                        put(i);
                end;
        l1[n]:=tong;
        for i:=n+1 to 2*n do
                begin
                        if (b[i]>b[q[1]]) and (top>0) then
                                begin
                                        j:=get;
                                        tong:=tong-b[j];
                                        tong:=tong+b[i];
                                        put(i);
                                end;
                        l1[i]:=tong;
                end;
        tong:=0;
        b[0]:=round(1e9)+1; q[0]:=0; top:=0;
        for i:=3*n downto 2*n+1 do
                begin
                        tong:=tong+b[i];
                        put1(i);
                end;
        l2[2*n+1]:=tong;
        for i:=2*n downto n do
                begin
                        if (b[i]<b[q[1]]) and (top>0) then
                                begin
                                        j:=get1;
                                        tong:=tong-b[j];
                                        tong:=tong+b[i];
                                        put1(i);
                                end;
                        l2[i]:=tong;
                end;
        kq:=-round(1e18);
        for i:=n to 2*n do if l1[i]-l2[i+1]>kq then kq:=l1[i]-l2[i+1];
        assign(g,fo); rewrite(g);
        writeln(g,kq);
        close(g);
end;

begin
        nhap;
       // traubo;
       xuli;
end.

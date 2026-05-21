var
        i,n,m,min,r1,r2,s:longint;
        x,y,z,a,b,id,lab:array [0..10000000] of longint;
procedure swap(var a,b:longint);
        var
                t:longint;
        begin
        t:=a;
        a:=b;
        b:=t;
        end;
procedure sort(l,r: longint);
      var
         i,j,x,y: longint;
      begin
         i:=l;
         j:=r;
         x:=a[(l+r) div 2];
         repeat
           while a[i]<x do
            inc(i);
           while x<a[j] do
            dec(j);
           if not(i>j) then
             begin
                swap(a[i],a[j]);
                swap(b[i],b[j]);
                swap(id[i],id[j]);
                inc(i);
                j:=j-1;
             end;
         until i>j;
         if l<j then
           sort(l,j);
         if i<r then
           sort(i,r);
      end;
procedure sort1(l,r: longint);
      var
         i,j,x,y: longint;
      begin
         i:=l;
         j:=r;
         x:=b[(l+r) div 2];
         repeat
           while b[i]<x do
            inc(i);
           while x<b[j] do
            dec(j);
           if not(i>j) then
             begin
                swap(a[i],a[j]);
                swap(b[i],b[j]);
                swap(id[i],id[j]);
                inc(i);
                j:=j-1;
             end;
         until i>j;
         if l<j then
           sort1(l,j);
         if i<r then
           sort1(i,r);
      end;
procedure sort2(l,r: longint);
      var
         i,j,x1: longint;
      begin
         i:=l;
         j:=r;
         x1:=z[(l+r) div 2];
         repeat
           while z[i]<x1 do
            inc(i);
           while x1<z[j] do
            dec(j);
           if not(i>j) then
             begin
                swap(x[i],x[j]);
                swap(y[i],y[j]);
                swap(z[i],z[j]);
                inc(i);
                j:=j-1;
             end;
         until i>j;
         if l<j then
           sort2(l,j);
         if i<r then
           sort2(i,r);
      end;
function getroot(v:longint):longint;
        begin
        while lab[v]>0 do v:=lab[v];
        getroot:=v;
        end;
procedure union(r1,r2:longint);
        var
                x:longint;
        begin
        x:=lab[r1]+lab[r2];
        if lab[r1]>lab[r2] then
                begin
                lab[r1]:=r2;
                lab[r2]:=x;
                end
        else
                begin
                lab[r1]:=x;
                lab[r2]:=r1;
                end;
        end;
begin
readln(n);
for i:=1 to n do
        begin
        readln(a[i],b[i]);
        id[i]:=i;
        end;
sort(1,n);
for i:=1 to n-1 do
        begin
        min:=a[i+1]-a[i];
        inc(m);
        x[m]:=id[i];
        y[m]:=id[i+1];
        z[m]:=min;
        end;
sort1(1,n);
for i:=1 to n-1 do
        begin
        min:=b[i+1]-b[i];
        inc(m);
        x[m]:=id[i];
        y[m]:=id[i+1];
        z[m]:=min;
        end;
for i:=1 to n do
        lab[i]:=-1;
sort2(1,m);
for i:=1 to m do
        begin
        r1:=getroot(x[i]);
        r2:=getroot(y[i]);
        if r1<>r2 then
                begin
                s:=s+z[i];
                union(r1,r2);
                end;
        end;
writeln(s);
end.
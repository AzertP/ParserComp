var n,w,k,i,j:longint;
    ans:int64;
    a,b,c:array[0..100001]of longint;  
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
                y:=a[i];
                a[i]:=a[j];
                a[j]:=y;
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
                y:=b[i];
                b[i]:=b[j];
                b[j]:=y;
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
         i,j,x,y:longint;
      begin
         i:=l;
         j:=r;
         x:=c[(l+r) div 2];
         repeat
           while c[i]<x do
            inc(i);
           while x<c[j] do
            dec(j);
           if not(i>j) then
             begin
                y:=c[i];
                c[i]:=c[j];
                c[j]:=y;
                inc(i);
                j:=j-1;
             end;
         until i>j;
         if l<j then
           sort2(l,j);
         if i<r then
           sort2(i,r);
      end;
begin
  readln(n);
  for i:=1 to n do 
    read(a[i]);
  for i:=1 to n do 
    read(b[i]);
  for i:=1 to n do 
    read(c[i]);
  sort(1,n);
  sort1(1,n);
  sort2(1,n);
  for i:=1 to n do 
    begin
      if b[1]>a[i]
      then k:=k+1;
    end;
  for i:=1 to n do 
    begin
      if c[i]>b[1]
      then w:=w+1;
    end;
  for i:=1 to n do 
    begin
      while (b[i]>a[k+1])and(k+1<=n) do k:=k+1;
      while (b[i]>=c[n-w+1])and(n-w+1<=n) do w:=w-1;
      ans:=ans+k*w;
    end;
  writeln(ans);
end.
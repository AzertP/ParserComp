const fi='';
      fo='';
      oo=trunc(1e18);
      maxn=trunc(1e5);
var   n:longint;
      sol:int64;
      nheapmin,nheapmax:longint;
      heapmax,heapmin:array[0..2*maxn]of longint;
      a:array[0..3*maxn]of longint;
      l,r:array[0..3*maxn]of int64;
function min(i,j:int64):int64; begin if i<j then exit(i); exit(j); end;
function max(i,j:int64):int64; begin if i>j then exit(i); exit(j); end;
procedure swap(var a,b:Longint);
var temp:longint;
begin
      temp:=a;
      a:=b;
      b:=temp;
end;
procedure upmax(i:longint);
begin
      if (i=1)or(heapmax[i div 2]>=heapmax[i]) then exit
      else begin
                  swap(heapmax[i],heapmax[i div 2]);
                  upmax(i div 2);
           end;
end;
procedure upmin(i:longint);
begin
      if (i=1)or(heapmin[i div 2]<=heapmin[i]) then exit
      else begin
                  swap(heapmin[i div 2],heapmin[i]);
                  upmin(i div 2);
           end;
end;
procedure downmax(i:longint);
var gt:longint;
begin
      gt:=i*2;
      if gt>nheapmax then exit;
      if (gt+1<=nheapmax)and(heapmax[gt]<heapmax[gt+1]) then inc(gt);
      if heapmax[i]>=heapmax[gt] then exit;
      begin
                    swap(heapmax[gt],heapmax[i]);
                    downmax(gt);
      end;
end;
procedure downmin(i:longint);
var gt:longint;
begin
      gt:=i*2;
      if gt>nheapmin then exit;
      if (gt+1<=nheapmin)and(heapmin[gt]>heapmin[gt+1]) then inc(gt);
      if heapmin[i]<=heapmin[gt] then exit;
      begin
                  swap(heapmin[gt],heapmin[i]);
                  downmin(gt);
      end;
end;
procedure pushmin(x:longint);
begin
       inc(nheapmin);
       heapmin[nheapmin]:=x;
       upmin(nheapmin);
end;
procedure pushmax(x:longint);
begin
      inc(nheapmax);
      heapmax[nheapmax]:=x;
      upmax(nheapmax);
end;
procedure popmin(x:longint);
begin
      heapmin[1]:=x;
      downmin(1);
end;
procedure popmax(x:longint);
begin
       heapmax[1]:=x;
       downmax(1);
end;
procedure inp;
var i:longint;
begin
      read(n);
      for i:=1 to 3*n do read(a[i]);
end;
procedure main;
var i,j:longint;
    res:int64;
    nmax,nmin:longint;
begin
      res:=0;
      for i:=1 to n do
      begin
          res:=res+a[i];
          pushmin(a[i]);
      end;
      l[n]:=res;
      for i:=n+1 to 2*n do
      begin
          if a[i]>heapmin[1] then
          begin
             res:=res-heapmin[1]+a[i];
             popmin(a[i]);
          end;
          l[i]:=res;
      end;
      res:=0;
      for i:=3*n downto 2*n+1 do
      begin
          res:=res+a[i];
          pushmax(a[i]);
      end;
      r[2*n+1]:=res;
      for i:=2*n downto n+1 do
      begin
          if a[i]<heapmax[1] then
          begin
             res:=res-heapmax[1]+a[i];
             popmax(a[i]);
          end;
          r[i]:=res;
      end;
      sol:=-oo;
      for i:=n to 2*n do sol:=max(sol,l[i]-r[i+1]);
      write(sol);
end;
begin
      assign(input,fi);
      reset(input);
      assign(output,fo);
      rewrite(output);
      inp;
      main;
      close(output);
end.

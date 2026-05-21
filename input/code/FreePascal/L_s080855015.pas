uses math;
var  heap,heap1:array[0..500000] of longint;
     res:int64;
     n,nheap:longint;
     f,g,a:array[0..500000] of int64;
procedure nhap;
var i:longint;
begin
 read(n);
 for i:=1 to 3*n do read(a[i]);
end;
procedure doicho(var x,y:longint);
var tg:longint;
begin
 tg:=x;
 x:=y;
 y:=tg;
end;
procedure upheap(i:longint);
var j:longint;
begin
 j:=i div 2;
 if (i=1) or (a[heap[i]]>a[heap[j]]) then exit;
 doicho(heap[i],heap[j]);
 upheap(j);
end;
procedure downheap(i:longint);
var j:longint;
begin
 j:=2*i;
 if j>nheap then exit;
 if (j<nheap) and (A[heap[j]]>a[heap[j+1]]) then inc(j);
 if a[heap[i]]>a[heap[j]] then
  begin
   doicho(heap[i],heap[j]);
   downheap(j);
  end;
end;
procedure push(x:longint);
begin
 inc(nheap);
 heap[nheap]:=x;
 upheap(nheap);
end;
function pop:longint;
begin
 pop:=heap[1];
 heap[1]:=heap[nheap];
 dec(nheap);
 downheap(1);
end;
procedure upheap1(i:longint);
var j:longint;
begin
 j:=i div 2;
 if (i=1) or (a[heap1[i]]<a[heap1[j]]) then exit;
 doicho(heap1[i],heap1[j]);
 upheap1(j);
end;
procedure downheap1(i:longint);
var j:longint;
begin
 j:=2*i;
 if j>nheap then exit;
 if (j<nheap) and (A[heap1[j]]<a[heap1[j+1]]) then inc(j);
 if a[heap1[i]]<a[heap1[j]] then
  begin
   doicho(heap1[i],heap1[j]);
   downheap1(j);
  end;
end;
procedure push1(x:longint);
begin
 inc(nheap);
 heap1[nheap]:=x;
 upheap1(nheap);
end;
function pop1:longint;
begin
 pop1:=heap1[1];
 heap1[1]:=heap1[nheap];
 dec(nheap);
 downheap1(1);
end;
procedure xuli;
var i,k:longint;
begin
 for i:=1 to n do
  begin
   push(i);
   f[i]:=f[i-1]+a[i];
  end;
 for i:=n+1 to 3*n do
  begin
   k:=pop;
   if a[i]>a[k] then
    begin
     f[i]:=f[i-1]-a[k]+a[i];
     push(i);
    end
   else
    begin
     f[i]:=f[i-1];
     push(k);
    end;
  end;
 nheap:=0;
 for i:=3*n downto 2*n+1 do
  begin
   push1(i);
   g[i]:=g[i+1]+a[i];
  end;
 for i:=2*n downto 1 do
  begin
   k:=pop1;
   if a[i]<a[k] then
    begin
     g[i]:=g[i+1]-a[k]+a[i];
     push1(i);
    end
   else
    begin
     g[i]:=g[i+1];
     push1(k);
    end;
  end;
 res:=-high(int64);
 for i:=n to 2*n do res:=max(Res,f[i]-g[i+1]);
 write(REs);
end;
begin
// assign(input,'b.inp');reset(input);
// assign(output,'b.out');rewrite(output);
 nhap;
 xuli;
// close(input); close(output);
end.

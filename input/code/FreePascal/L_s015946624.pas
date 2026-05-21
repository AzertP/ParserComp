const   fi='';//B.inp';
        fo='';//B.out';
var     f:text;
        i,j,n,top,top1:longint;
        kq:int64;
        heap,heap1,a:array[0..5000000] of longint;
        l,g:array[0..5000000] of int64;
(*===================================*)
procedure inp;
Begin
assign(f,fi);
reset(f);
readln(f,n);
for i:=1 to 3*n do read(f,a[i]);
close(f);
End;
(*===================================*)
procedure swap(var a,b:longint);
var y:longint;
Begin
y:=a;
a:=b;
b:=y;
End;
(*===================================*)
procedure upheap(i:longint);
Begin
if (i=1) or (heap[i]<heap[i div 2]) then exit;
swap(heap[i],heap[i div 2]);
upheap(i div 2);
End;
(*===================================*)
procedure downheap(i:longint);
var j:longint;
Begin
j:=i*2;
if j>top then exit;
if (j<top) and (heap[j]<heap[j+1]) then inc(j);
if heap[i]<heap[j] then
 begin
  swap(heap[i],heap[j]);
  downheap(j);
 end;
End;
(*===================================*)
procedure push(x:longint);
Begin
inc(top);
heap[top]:=x;
upheap(top);
End;
(*===================================*)
function pop:longint;
Begin
pop:=heap[1];
heap[1]:=heap[top];
dec(top);
downheap(1);
End;
(*===================================*)
procedure upheap1(i:longint);
Begin
if (i=1) or (heap1[i]>heap1[i div 2]) then exit;
swap(heap1[i],heap1[i div 2]);
upheap1(i div 2);
End;
(*===================================*)
procedure downheap1(i:longint);
var j:longint;
Begin
j:=i*2;
if j>top1 then exit;
if (j<top1) and (heap1[j]>heap1[j+1]) then inc(j);
if heap1[i]>heap1[j] then
 begin
  swap(heap1[i],heap1[j]);
  downheap1(j);
 end;
End;
(*===================================*)
procedure push1(x:longint);
Begin
inc(top1);
heap1[top1]:=x;
upheap1(top1);
End;
(*===================================*)
function pop1:longint;
Begin
pop1:=heap1[1];
heap1[1]:=heap1[top1];
dec(top1);
downheap1(1);
End;
(*===============================*)
procedure trong;
Begin
for i:=1 to n do begin l[i]:=l[i-1]+a[i]; push1(a[i]); end;
for i:=n+1 to 3*n do
  begin
   l[i]:=l[i-1]+a[i];
   push1(a[i]);
   l[i]:=l[i]-pop1;
  end;

for i:=3*n downto 2*n+1 do begin g[i]:=g[i+1]+a[i]; push(a[i]); end;
for i:=2*n downto 1 do
 begin
  g[i]:=g[i+1]+a[i];
  push(a[i]);
  g[i]:=g[i]-pop;
 end;
End;
(*===============================*)
BEGIN
inp;
assign(f,fo);
rewrite(f);
trong;
kq:=-maxlongint*100000;
for i:=n to 2*n do
 if l[i]-g[i+1]>kq then kq:=l[i]-g[i+1];
writeln(f,kq);
close(f);
END.
type
  pair = record
    a : longint;
    b : longint;
  end;
	pairArray = array[1..100000] of pair;

var i,j,n,m:longint;
    ans:int64;
    input:pairArray;
    heap:array[0..100000] of longint;

procedure swap(var X, Y: longint);
begin
	if X <> Y then begin
		X := X xor Y;
		Y := X xor Y;
		X := X xor Y
	end
end;

procedure	pswap(var a,b:pair);
var tmp:pair;
begin
	tmp:=a;
	a:=b;
	b:=tmp
end;

procedure QuickSort(var a:pairArray;start_index,end_index:longint);
var i,j,x:int64;
begin
	x:=a[(start_index+end_index)div 2].a;
	i:=start_index;
	j:=end_index;
	while(true)do
	begin
		while(a[i].a<x)do i:=i+1;
		while(x<a[j].a)do j:=j-1;
		if(i<j)then
		begin
			pswap(a[i],a[j]);
			i:=i+1;
			j:=j-1;
		end else
		begin
			if(start_index<i-1)then QuickSort(a,start_index,i-1);
			if(j+1<end_index)then QuickSort(a,j+1,end_index);
			exit;
		end;
	end;
end;

procedure upheap(b:longint);
var p:integer;
begin
  heap[0] := heap[0] + 1;
  p := heap[0];
  heap[p] := b;
  while (p>1) and (heap[p]>heap[p div 2]) do begin
    swap(heap[p],heap[p div 2]);
    p := p div 2
  end
end;

function downheap:longint;
var p:integer;
begin
  downheap:=heap[1];
  heap[1] := heap[heap[0]];
  heap[0] := heap[0] - 1;
  p := 1;
  while (p*2 <= heap[0]) and (heap[p]<heap[p*2]) or (p*2+1 <= heap[0]) and (heap[p]<heap[p*2+1]) do
    if (p*2+1 <= heap[0]) and (heap[p*2]<heap[p*2+1]) then begin
      swap(heap[p],heap[p*2+1]);
      p := p*2+1
    end else begin
      swap(heap[p],heap[p*2]);
      p := p*2
    end
end;

begin
  readln(n,m);
  for i := 1 to n do readln(input[i].a,input[i].b);
  QuickSort(input,1,n);
  j := 1;
  ans := 0;
  for i := 1 to m do begin
    while input[j].a=i do begin
      upheap(input[j].b);
      j := j + 1
    end;
    if heap[0]>0 then ans := ans + downheap
  end;
  writeln(ans)
end.
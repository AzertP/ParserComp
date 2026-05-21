const
        fi='arc074_b.inp';
        fo='arc074_b.out';
var
        res:int64;
        n,nheap:longint;
        P1,P2:Array[0..300001] of int64;
        A:Array[0..300001] of longint;
        Heap:Array[0..100001] of longint;
{------------------------------------}
procedure swap(var xx1,xx2:longint);
var tg:longint;
        begin
                tg:=xx1;
                xx1:=xx2;
                xx2:=tg;
        end;
{------------------------------------}
procedure upheapmin(k:longint);
        begin
                if (k=1) or (Heap[k]>=Heap[k div 2]) then exit
                else
                        begin
                                swap(Heap[k],Heap[k div 2]);
                                upheapmin(k div 2);
                        end;
        end;
{------------------------------------}
procedure downheapmin(k:longint);
var vtk:longint;
        begin
                if (2*k<=nheap) then
                begin
                        if (2*k+1<=nheap) and (Heap[2*k+1]<Heap[2*k]) then vtk:=2*k+1 else vtk:=2*k;
                        if Heap[k]>Heap[vtk] then
                        begin
                                swap(Heap[k],Heap[vtk]);
                                downheapmin(vtk);
                        end;
                end;
        end;
{------------------------------------}
procedure pushmin(pt:longint);
        begin
                inc(nheap);
                Heap[nheap]:=pt;
                upheapmin(nheap);
        end;
{------------------------------------}
procedure upheapmax(k:longint);
        begin
                if (k=1) or (Heap[k]<=Heap[k div 2]) then exit
                else
                        begin
                                swap(Heap[k],Heap[k div 2]);
                                upheapmax(k div 2);
                        end;
        end;
{------------------------------------}
procedure downheapmax(k:longint);
var vtk:longint;
        begin
                if (2*k<=nheap) then
                begin
                        if (2*k+1<=nheap) and (Heap[2*k+1]>Heap[2*k]) then vtk:=2*k+1 else vtk:=2*k;
                        if Heap[k]<Heap[vtk] then
                        begin
                                swap(Heap[k],Heap[vtk]);
                                downheapmax(vtk);
                        end;
                end;
        end;
{------------------------------------}
procedure pushmax(pt:longint);
        begin
                inc(nheap);
                Heap[nheap]:=pt;
                upheapmax(nheap);
        end;
{------------------------------------}
procedure main;
var o:longint;
        begin
                read(n);
                for o:=1 to 3*n do read(A[o]);
                nheap:=0;
                for o:=1 to n do
                begin
                        pushmin(A[o]);
                        inc(P1[n],A[o]);
                end;
                for o:=n+1 to 2*n do
                begin
                        P1[o]:=P1[o-1];
                        if A[o]>Heap[1] then
                        begin
                                P1[o]:=P1[o]-Heap[1]+A[o];
                                Heap[1]:=A[o];
                                downheapmin(1);
                        end;
                end;
                nheap:=0;
                for o:=3*n downto 2*n+1 do
                begin
                        pushmax(A[o]);
                        inc(P2[2*n+1],A[o]);
                end;
                for o:=2*n downto n+1 do
                begin
                        P2[o]:=P2[o+1];
                        if A[o]<Heap[1] then
                        begin
                                P2[o]:=P2[o]-Heap[1]+A[o];
                                Heap[1]:=A[o];
                                downheapmax(1);
                        end;
                end;
                res:=-1000000000000000;
                for o:=n to 2*n do if res<P1[o]-P2[o+1] then res:=P1[o]-P2[o+1];
                write(res);
        end;
{------------------------------------}
begin
//        assign(input,fi); reset(input);
  //      assign(output,fo);rewrite(output);
        main;
end.

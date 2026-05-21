var cc : array [-1..100001] of longint ;
    max,n,a,b,c,d,e,f,g : longint ;
begin
 readln(n);
 for a:=1 to n do
  begin
   read(g);
   cc[g-1]:=cc[g-1]+1;
   cc[g]:=cc[g]+1;
   cc[g+1]:=cc[g+1]+1;
  end;
 for a:=-1 to 100001 do
  if cc[a]>max then
   max:=cc[a];
 writeln(max); 
end.

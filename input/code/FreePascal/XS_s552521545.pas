var a,b,c,x,i,j,k,ans:longint;
begin
  read(a,b,c,x);
for i:=0 to a do for j:=0 to b do for k:=0 to c do if (500*i+100*j+50*k=x) then inc(ans);writeln(ans);end.

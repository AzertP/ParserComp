var a,b,i,j,n,ans:longint;
var s:string;
begin
  readln(s);readln(n);
  for i:=1 to length(s) do 
  begin
    if (1+n*(i-1)>length(s))then break;
    write(s[1+n*(i-1)]);
  end;
  writeln;
end.
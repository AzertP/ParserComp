var s:string;
    i,j,k:longint;
    x,y,ans:longint;
begin
  readln(s);
  for i:=1 to length(s) do
  begin
    if(x=y)then
    begin
      x:=x+1;
      if(s[i]='p')then
      ans:=ans-1;
    end
    else
    begin
      y:=y+1;
      if(s[i]='g')then
      ans:=ans+1;
    end;
  end;
  writeln(ans);
end.
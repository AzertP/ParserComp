var
  s:string;
  i,p:longint;
  ok:boolean;
begin
  readln(s);
  for i:=1 to length(s) do
  if s[i]='C' then
  begin
    ok:=true;
    p:=i;
    break;
  end;
  if not ok then 
  begin
    writeln('No');
    exit;
  end;
  ok:=false;
  for i:=p+1 to length(s) do
  if s[i]='F' then ok:=true;
  if ok then writeln('Yes')
        else writeln('No');
end.
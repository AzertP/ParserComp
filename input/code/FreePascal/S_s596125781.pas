var a  :char;
    s,t  :array[0..200010]of integer;
    n,i,j,k,ans  :integer;
    
function min(a,b:integer):integer;
begin
  if a>b then min:=b
  else min:=a
end;

begin
  readln(n);
  s[0]:=0;
  for i:=1 to n do begin
    read(a);
    if a='#' then s[i]:=s[i-1]+1
    else s[i]:=s[i-1];
    end;
  ans:=210000;
  for i:=0 to n do t[i]:=s[i]+n+s[i]-i-s[n];
  for i:=0 to n do ans:=min(ans,t[i]);
  writeln(ans)
end.
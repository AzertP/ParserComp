var
  n:string;
  a:array['a'..'z'] of 0..1;
  s,i:longint;
  ch:char;
begin
  readln(n);
  for i:=1 to length(n) do a[n[i]]:=1;
  for ch:='a' to 'z' do s:=s+a[ch];
  if s=26 then begin writeln('None');exit;end;
  for ch:='a' to 'z' do if a[ch]=0 then begin writeln(ch);exit;end;
end.
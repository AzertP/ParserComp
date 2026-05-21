var n,m,k,a,i,j,ans  :integer;
    s:array[1..30]of integer;
begin
  readln(n,m);
  for i:=1 to 30 do s[i]:=0;
  for i:=1 to n do begin
    read(k);
    for j:=1 to k do begin
      read(a);
      s[a]:=s[a]+1
    end;
    readln
  end;
  for i:=1 to m do if s[i]=n then ans +=1;
  write(ans)
  end.

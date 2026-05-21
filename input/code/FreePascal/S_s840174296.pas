var x,y,i,j,cou,k:integer;
m:array[1..100,1..100]of char;
begin 
  readln(x,y);
  for i:=1 to x do 
  begin 
    cou:=0;
    for j:=1 to y do 
       begin read(m[i,j]);if m[i,j]='.' then inc(cou);end;
    if cou=y then for k:=1 to y do m[i,k]:='*';
    readln;
  end;
  for i:=1 to y do
  begin
    cou:=0;
    for j:=1 to x do 
      if m[j,i]='#' then inc(cou);
    if cou=0 then for k:=1 to x do m[k,i]:='*';
  end;
  for i:=1 to x do 
    begin
		cou:=0;
		for j:=1 to y do
		if m[i,j]<>'*' then begin write(m[i,j]);cou:=1;end;
		if cou=1 then writeln; 
    end;
end.
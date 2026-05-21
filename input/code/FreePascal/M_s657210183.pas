var
  s1,s2:string;
  a,b:array['a'..'z']of longint;
  c:array[1..26]of boolean;
  f:boolean;
  i,j,k,p:longint;
begin
  readln(s1);
  read(s2);
  k:=length(s2);
  f:=true;
  if length(s1)<length(s2) then begin k:=length(s1); f:=false; end;
  for i:=1 to k-1 do
  begin
    inc(a[s1[i]]);
	inc(b[s2[i]]);
  end;
  if f then inc(b[s2[k]])
  else inc(a[s1[k]]);
  if f then p:=length(s1)
  else p:=length(s2);
  for i:=k to p do
  begin
  begin
     if f then inc(a[s1[i]])
	 else inc(b[s2[i]]);
  end;
  end;
  f:=true;
  for i:=1 to 26 do
  begin
     p:=a[chr(i+ord('a')-1)];
	 for j:=1 to 26 do
	 begin
	    if (c[j]=false)and(b[chr(j+ord('a')-1)]=p) then 
		begin
		   c[j]:=true;
		   break;
		end;
		if j=26 then f:=false;
     end;
	 if f=false then begin writeln('No'); halt; end;
  end;
  writeln('Yes');
end.

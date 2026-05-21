var s,s2:ansistring;
pos1,pos2,pos3,pos4,cnt,i:longint;
begin
  readln(s);
  for i:=length(s) downto 1 do
    s2:=s2+s[i];
  cnt:=1;
  pos1:=0;
  pos2:=0;
  pos3:=0;
  pos4:=0;
  while (true) do
    begin
      pos1:=pos('maerd',s2);
      pos2:=pos('remaerd',s2);
      pos3:=pos('esare',s2);
      pos4:=pos('resare',s2);
      //writeln(s2);
      if (length(s2)=0) then break
      else if (pos1=0) and (pos2=0) and (pos3=0) and (pos4=0) then
        begin
          writeln('NO');
          halt;
        end
      else if (pos1=1) and ((pos2=0) or (pos1<pos2)) and ((pos3=0) or (pos1<pos3)) and ((pos4=0) or (pos1<pos4)) then
        begin
          delete(s2,1,5);
          //writeln('1');
        end
        else if (pos2=1) and ((pos1=0) or (pos2<pos1)) and ((pos3=0) or (pos2<pos3)) and ((pos4=0) or (pos2<pos4)) then
        begin
          delete(s2,1,7);
          //writeln('2');
        end
        else if (pos3=1) and ((pos1=0) or (pos3<pos1)) and ((pos2=0) or (pos3<pos2)) and ((pos4=0) or (pos3<pos4)) then
        begin
          delete(s2,1,5);
          //writeln('3');
        end
        else if (pos4=1) and ((pos1=0) or (pos4<pos1)) and ((pos2=0) or (pos4<pos2)) and ((pos3=0) or (pos4<pos3)) then
        begin
          delete(s2,1,6);
          //writeln('4');
        end
        else begin
        if length(s2)=0 then break;
          writeln('NO');
          halt;
        end;
    end;
    writeln('YES');
end.

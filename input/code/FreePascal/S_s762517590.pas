var n,sl:qword;

begin
  //Assign(input,'a.inp');  reset(input);
  //Assign(output,'a.out');  rewrite(output);

  read(n);
  if(n mod 2=0) then
    begin
      sl:=0;
      n:=n div 2;
      while(n<>0) do
        begin
          sl:=sl+(n div 5);
          n:=n div 5;
        end;
      write(sl);
    end
  else write(0);
end.

program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  a,h,w,i,j,count,x,y,sw,m,n,b,c,soeji,sum,max,k,min:Integer;
  l:array of Integer;
  s,s123,s1,s2,s3:string;
  al:string = 'abcdefghijklmnopqrstuvwxyz';
  tf,tfl,tfr:Boolean;
  so:array[0..24] of integer = (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97);
begin
  try
    { TODO -oUser -cConsole メイン : ここにコードを記述してください }
    Readln(n);
    count:=0;
    max  :=0;
    min  :=100;

    for j:=1 to n do begin //sagasu
      count:=0;
      if j mod 2 <> 0 then begin
        for i := 1 to 35 do //kazoeru
          if j mod (i*2-1) = 0 then
            inc(count);
        if count=7 then
          inc(max);
      end;
    end;

    Writeln(max);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  w,h,n,ni,i,j,a,x,y,sum:Integer;
  xy:array of array of Integer;
begin
  try
    { TODO -oUser -cConsole Main : ぬりえ }
    Read(w);
    Read(h);
    Readln(n);

    SetLength(xy,h,w);

    for i := 0 to h - 1 do
      for j := 0 to w - 1 do
        xy[i,j]:=0;

    for ni := 0 to n-1 do begin
      Read(x);
      Read(y);
      Readln(a);
      case a of
        1:begin
          for i := 0 to h - 1 do
            for j := 0 to x - 1 do
              xy[i,j]:=1;
        end;
        2:begin
          for i := 0 to h-1 do
            for j := x to w - 1 do
              xy[i,j]:=1;
        end;
        3:begin
          for i := 0 to y - 1 do
            for j := 0 to w - 1 do
              xy[i,j]:=1;
        end;
        4:begin
          for i := y to h - 1 do
            for j := 0 to w - 1 do
              xy[i,j]:=1;
        end;
      end;
    end;

    sum:=0;
    for i := 0 to h - 1 do
      for j := 0 to w - 1 do
        sum:=sum+(xy[i,j]xor 1 xor 0);

    Writeln(sum);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.
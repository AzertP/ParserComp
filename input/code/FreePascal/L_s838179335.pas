program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  n,i,j,temp,min:Integer;
  d:array of Integer;
begin
  try
    { TODO -oUser -cConsole Main : ここにコードを記述してください }
    Readln(n);
    SetLength(d,n);
    for i:=0 to n-1 do
      Readln(d[i]);

    for i := 0 to n-1 do begin
      min := i;
      for j := i+1 to n-1 do begin
        if d[j] < d[min] then
        min := j;
      end;
      temp:=d[i];
      d[i]:=d[min];
      d[min]:=temp;
    end;

    for i := 0 to n - 2 do begin
      if d[i]=d[i+1] then
        n:=n-1;
    end;

    Writeln(n);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  n,i,j,temp,min:Integer;
  d:array of Integer;
begin
  try
    { TODO -oUser -cConsole Main : ここにコードを記述してください }
    Readln(n);
    SetLength(d,n);
    for i:=0 to n-1 do
      Readln(d[i]);

    for i := 0 to n-1 do begin
      min := i;
      for j := i+1 to n-1 do begin
        if d[j] < d[min] then
        min := j;
      end;
      temp:=d[i];
      d[i]:=d[min];
      d[min]:=temp;
    end;

    for i := 0 to n - 2 do begin
      if d[i]=d[i+1] then
        n:=n-1;
    end;

    Writeln(n);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.
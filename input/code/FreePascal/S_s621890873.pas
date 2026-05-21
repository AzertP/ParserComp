program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  m:integer;
begin
  try
    { TODO -oUser -cConsole Main : abc84a}
    readln(m);
    Writeln(48-m);
    readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

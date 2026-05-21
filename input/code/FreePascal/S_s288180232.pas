program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils,Classes;
var
  h,w,i,j,count,x,y,sw,m,n,c:Integer;
  a:array[0..12] of Integer = (0,1,3,1,2,1,2,1,1,2,1,2,1);
  s123,s1,s2,s3:string;
  tf:Boolean;
  nyu:TStringList;
begin
  try
    { TODO -oUser -cConsole Main : ここにコードを記述してください }
    Read(x);
    Readln(y);

    if a[x]=a[y] then
      Writeln('Yes')
    else
      Writeln('No');
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

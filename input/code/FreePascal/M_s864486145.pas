program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  a,h,w,i,j,count,x,y,sw,m,n,b,c,soeji,sum,max,min,k:Integer;
  l:array of char;
  s,s123,s1,s2,s3:string;
  tf:Boolean;
begin
  try
    { TODO -oUser -cConsole Main : ここにコードを記述してください }
    Read(a);
    Read(b);
    Read(c);
    Read(x);
    Readln(y);

    count:=0;
    min:=(x+y-abs(x-y))div 2;

    if ((a+b)/2)>c then begin
      count:=2*c*min;
      x:=x-min;
      y:=y-min;
    end;

    if a<(2*c) then
      count:=count+a*x
    else
      count:=count+2*c*x;

    if b<(2*c) then
      count:=count+b*y
    else
      count:=count+2*c*y;


    Writeln(count);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.
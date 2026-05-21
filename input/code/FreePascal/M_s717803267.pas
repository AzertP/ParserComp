program Project1;
     
{$APPTYPE CONSOLE}
     
uses
  SysUtils;
var
  a,h,w,i,j,count,x,y,sw,m,n,b,c,soeji,sum,max,k,min,ax,ay,bx,by:Integer;
  l:array of Integer;
  s,s123,s1,s2,s3:string;
  al:string = 'abcdefghijklmnopqrstuvwxyz';
  tf,tfl,tfr:Boolean;
begin
  try
    { TODO -oUser -cConsole メイン : ここにコードを記述してください }
    Read  (ax);
    Read  (ay);
    Read  (bx);
    Readln(by);
    count:=0;
    max  :=0;
    min  :=100;

    Write  (bx-(by-ay));
    Write  (' ');
    Write  (by+(bx-ax));
    Write  (' ');
    Write  (ax+ay-by);
    Write  (' ');
    Writeln(ay-(ax-bx));
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

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
begin
  try
    { TODO -oUser -cConsole メイン : ここにコードを記述してください }
    Readln(a);
    count:=0;
    max  :=0;
    min  :=100;

    if (a=1 )or
       (a=2 )or
       (a=3 )or
       (a=5 )or
       (a=6 )or
       (a=9 )or
       (a=10)or
       (a=13)or
       (a=17)  then
      Writeln('No')
    else
      Writeln('Yes');
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

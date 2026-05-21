program Project1;

{$APPTYPE CONSOLE}

uses
  SysUtils;
var
  l,r,a,h,w,i,j,count,x,y,sw,m,n,b,c,soeji,sum,max,k,t:Integer;
  //l:array of Integer;
  s,s123,s1,s2,s3:string;
  tf:Boolean;
  //nyu:TStringList;
begin
  try
    { TODO -oUser -cConsole Main : ここにコードを記述してください }
    Readln(n);

    sum:=0;
    for i := 0 to n - 1 do begin
      Read(l);
      Readln(r);
      sum:=sum+r-l+1;
    end;
    Writeln(sum);
    Readln;
  except
    on E: Exception do
      Writeln(E.ClassName, ': ', E.Message);
  end;
end.

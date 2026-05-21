var a:array[0..100] of longint;
var them,n:longint;

procedure xuly(x:longint);
begin
   if (1<=x) and (x<=399) then inc(a[1])
      else if (x<=799) then inc(a[2])
         else if (x<=1199) then inc(a[3])
            else if (x<=1599) then inc(a[4])
               else if (x<=1999) then inc(a[5])
                  else if (x<=2399) then inc(a[6])
                     else if (x<=2799) then inc(a[7])
                        else if (x<=3199) then inc(a[8])
                           else inc(them);
end;

procedure nhap;
var i,the:Longint;
begin
   read(n);
   fillchar(a,sizeof(a),0);
   them:=0;
   for i:=1 to n do
      begin
      read(the);
      xuly(the);
      end;
end;

procedure xuat;
var i,tong:longint;
begin
   tong:=0;
      for i:=1 to 8 do
         if a[i]>0 then inc(tong);

   if tong=0 then write('1',' ') else
      write(tong,' ');
   write(tong+them);
end;

begin
   nhap;
   xuat;
end.
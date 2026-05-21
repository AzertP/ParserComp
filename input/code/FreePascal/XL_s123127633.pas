{你无权查看本记录 因为↓}

















































{你是傻逼
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
    SBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBSBS
}



































{
    祝您一路顺风，半路失踪，回家发疯~~~~~~~~~~~~
}










































{再见了，SB，我们后会无期}



































{别以为我不知道是你YMJ！！！，喜欢看人家程序的**→_→}























































































var
 x,i,j:longint;
 a:array[0..5,0..5] of longint;
begin
 readln(a[1,1],a[1,2],a[2,2]);
 x:=a[2,2]*3;
 a[1,3]:=x-a[1,1]-a[1,2];
 a[3,1]:=x-a[1,3]-a[2,2];
 a[3,3]:=x-a[1,1]-a[2,2];
 a[2,1]:=x-a[1,1]-a[3,1];
 a[2,3]:=x-a[2,1]-a[2,2];
 a[3,2]:=x-a[1,2]-a[2,2];
 for i:=1 to 3 do
  begin
   for j:=1 to 3 do
    write(a[i,j],' ');
   writeln;
  end; 
end.
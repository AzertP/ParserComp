var
    a,b,c,i,j,z:longint;
    s:array[1..10000]of string;//仅用来判断位数
    k:array[1..10000]of longint;//记录数字（10^4=10000）
    ans:qword;//要开大点，如果是1+……+10000用int64会爆
begin
    read(a,b,c);
    for i:=1 to a do
    begin
        str(i,s[i]);//把数字转为字符串
        for j:=1 to length(s[i]) do//判断位数
        begin
            val(s[i][j],z);//一个个数字枚举
            k[i]:=k[i]+z;//把每一位数加起来
        end;
        if(k[i]>=b)and(k[i]<=c)then inc(ans,i);//判断
    end;
    writeln(ans);//输出
end.
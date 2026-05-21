var     n:longint;
        a,b:array[1..1000]of longint;

procedure nhap;
var     i:longint;
begin
        readln(n);
        for i:=1 to n do read(a[i]);
end;

procedure xuli;
var     i:longint;
        dem,dem1:longint;
begin
        for i:=1 to n do
            begin
                if (a[i]>=1) and (a[i]<400) then inc(b[1]);
                if (a[i]>=400) and (a[i]<800) then inc(b[2]);
                if (a[i]>=800) and (a[i]<1200) then inc(b[3]);
                if (a[i]>=1200) and (a[i]<1600) then inc(b[4]);
                if (a[i]>=1600) and (a[i]<2000) then inc(b[5]);
                if (a[i]>=2000) and (a[i]<2400) then inc(b[6]);
                if (a[i]>=2400) and (a[i]<2800) then inc(b[7]);
                if (a[i]>=2800) and (a[i]<3200) then inc(b[8]);
                if a[i]>=3200 then inc(b[9]);
            end;
        dem:=0;
        for i:=1 to 8 do
            if b[i]>0 then inc(dem);
        if b[9]=0 then writeln(dem,' ',dem)
        else
            begin
                dem1:=dem+b[9];
                if dem=0 then  write('1 ',dem1)
                else writeln(dem,' ',dem1)
            end;
end;

procedure test;
var     i:longint;
begin
        assign(output,'at_03.inp');rewrite(output);
                randomize;
                n:=random(10);
                writeln(n+1);
                for i:=1 to n+1 do
                        write(random(4000)+1,' ');
        close(output);
end;

begin
        //test;
        //assign(input,'at_03.inp');reset(input);
        //assign(output,'at_04.out');rewrite(output);
                nhap;
                xuli;
end.


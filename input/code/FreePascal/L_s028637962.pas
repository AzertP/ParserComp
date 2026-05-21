//{$R-,Q-,S-,I-}
{$OPTIMIZATION LEVEL2}
{$INLINE ON}

uses math;

procedure mofile;
begin
        {$IFDEF ONLINE_JUDGE}
        assign(input,'');
        reset(input);
        assign(output,'');
        rewrite(output);
        {$ELSE}
        assign(input,'checking.inp');
        reset(input);
        assign(output,'');
        rewrite(output);
        {$ENDIF}
end;

procedure dongfile;
begin
        close(input);
        close(output);
end;

const
        maxc=1000000000+69;
        modulo=1000000000+7;
        maxn=400000+10;

type
        re=record
                x,y:longint;
        end;

var
        n,q,dem,res:longint;
        s:array[0..maxn] of ansistring;
        a,c:array[0..maxn*4,'`'..'z'] of longint;
        cnt:array[0..maxn,'`'..'z','`'..'z'] of longint;
        al:ansistring;

procedure insert(p,x,i:longint);
begin
        if x=length(s[i])+1 then exit;
        inc(c[p][s[i,x]]);
        if a[p][s[i,x]]>0 then insert(a[p][s[i,x]],x+1,i) else
        begin
                inc(dem);
                a[p][s[i,x]]:=dem;
                insert(dem,x+1,i);
        end;
end;

procedure nhapdl;
var
        i:longint;
begin
        readln(n);
        for i:=1 to n do
        begin
                readln(s[i]);
                s[i]:=s[i]+'`';
                insert(0,1,i);
        end;
end;

procedure dfs(p,x,i:longint);
var
        j:char;
begin
        if x=length(s[i])+1 then exit;
        for j:='`' to 'z' do
        begin
                inc(cnt[i,j,s[i,x]],c[p,j]);
        end;
        dfs(a[p][s[i,x]],x+1,i);
end;

procedure xuli;
var
        o,i,j,x:longint;
        blank:char;
begin
        for i:=1 to n do dfs(0,1,i);
        readln(q);
        for o:=1 to q do
        begin
                readln(x,blank,al);
                al:='`'+al;

                res:=0;
                for i:=1 to 26 do
                for j:=i+1 to 27 do
                res:=res+cnt[x,al[i],al[j]];

                writeln(res+1);
        end;
end;

begin
        //mofile;
        nhapdl;
        xuli;
        dongfile;
end.













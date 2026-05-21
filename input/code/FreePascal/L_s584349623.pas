        {$mode objfpc}
        {$coperators on}
program Lexicographical_disorder________________________CODE_FESTIVAL_2016_qual_B____________________________________________________________;
const
        fi = 'lex.inp';
        fo = 'lex.out';
        sz=round(1e5)+100;
type
    Tnode = ^node;
    node  =
        record
            c:array['_'..'z'] of Tnode;
            cnt:longint;
        end;
var
    n,q,kth:longint;
    a:array[1..sz]of ansistring;
    s:string;
    nod,root:Tnode;
    count:array[0..sz,'_'..'z','_'..'z']of longint;


function min(const x,y:longint):longint; begin if x<=y then exit(x) else exit(y); end;
function max(const x,y:longint):longint; begin if x>=y then exit(x) else exit(y); end;
procedure spaw(var nod:Tnode);
var i:char;
    begin
        new(nod);
        nod^.cnt:=0;
        for i:='_' to 'z' do nod^.c[i]:=nil;
    end;

procedure enter;
var i : integer;
    begin
        readln(n);
        for i:=1 to n do
            begin
                readln(a[i]);
                a[i]:=a[i]+'_';
            end;
        readln(q);
    end;
procedure update(const a:ansistring);
var ch,rr:char;
    i:longint;
    begin
        nod:=root;
        for i:=1 to length(a) do
            begin
                ch:=a[i];
                if nod^.c[ch]=nil then spaw(nod^.c[ch]);
                nod := nod^.c[ch];
                inc(nod^.cnt);
            end;
    end;
procedure updateElement(p:longint);
var i:longint;
    ch,rr:char;
    begin
        nod:=root;
        for i:=1 to length(a[p]) do
            begin
                ch:=a[p][i];
                for rr:='_' to 'z' do if (rr<>ch)and(nod^.c[rr]<>nil) then
                    count[p][ch][rr] += nod^.c[rr]^.cnt;
                nod:=nod^.c[ch];
            end;
    end;
procedure Build();
var i:longint;
    begin
        spaw(root);
        for i:=1 to n do
            begin
                update(a[i]);
            end;
        for i:=1 to n do
            begin
                updateElement(i);
            end;
    end;
procedure doQuerry;
var i,j:longint;
    ans:longint;
    begin
        ans:=0;
        for i:=1 to 26 do
            for j:=1 to i-1 do ans += count[kth,s[i],s[j]];
        for i:=1 to 26 do ans += count[kth,s[i],'_'];
        ans+=1;
        writeln(ans);
    end;
procedure solve;
var i: integer;
    begin
        Build();
        for i:=1 to q do
            begin
                read(kth);
                readln(s); delete(s,1,1);
                doQuerry;
            end;
    end;
procedure print();
var i:longint;
    begin
    end;
begin
        enter(); solve(); print();
end.

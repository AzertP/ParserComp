//uses crt;
var n,i,o,p,k,l,j,m,x,z,count,max:longint;
used:array[1..16] of boolean;
type persoana = record
        testi:integer;
        saidabout:array[1..16] of integer;
        heis:array[1..16] of boolean;
        sincere:boolean;
        someonesaid:boolean;
        used:boolean;
        end;
var om:array[1..16] of persoana;
procedure backtrack(step:longint);
var i,o,p,k,j,curent:longint;
contradiction:boolean;
begin
if step=n+1 then begin
        contradiction:=false;
        k:=0;
        for i:=1 to n do
                om[i].sincere:=false;
                om[i].someonesaid:=false;
        for i:=1 to n do
                if used[i] then begin
                        inc(k);
                        om[i].sincere:=true;
                        om[i].someonesaid:=true;
                        //write(i,' ');
                        end;
               // writeln;
        for i:=1 to n do begin
                if contradiction then break;
                if om[i].sincere then
                        for j:=1 to om[i].testi do begin
                                curent:=om[i].saidabout[j];
                              //  writeln(curent);
                                if (om[curent].someonesaid) and (om[i].heis[j] <> om[curent].sincere) then begin
                                      //  writeln('omul',curent,' nu a zis nimeni si omul', i,' spune ca el este ',om[i].heis[j],' ceea ce este diferit de ',om[curent].sincere);
                                        contradiction:=true;
                                        break;
                                        end;
                                if (om[curent].someonesaid=false) and (om[curent].sincere <> om[i].heis[j]) then begin
                                        om[curent].sincere:=om[i].heis[j];
                                      //  writeln(curent);
                                        inc(K);
                                        end;
                                        end;
                                end;
      //  writeln(k,contradiction);
        if not contradiction then if k>max then max:=k;
        end
else begin
        used[step]:=true;
        backtrack(step+1);
        used[step]:=false;
        backtrack(step+1);
        end;
end;
begin
read(n);
for i:=1 to n do begin
        read(l);
        for j:=1 to l do begin
                read(x,z);
                om[i].saidabout[j]:=x;
                if z=1 then om[i].heis[j]:=true
                else om[i].heis[j]:=false;
              //  writeln(om[i].saidabout[j],' ',om[i].heis[j]);
                om[i].testi:=l;
                end;
        end;
backtrack(1);
writeln(max);
readln;
readln;
end.

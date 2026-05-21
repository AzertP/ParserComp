const fi='STONE.INP';
      fo='STONE.OUT';
      maxn=100;
      maxk=trunc(1e5);

var n,k:longint;
    a:array[1..maxn] of longint;
    f:array[0..maxk] of 0..1; {f=0-> thua, f=1->thang}
//============================
procedure doc;
var i:longint;
begin
  readln(n,k);

  for i:=1 to n do read(a[i]);
end;
//============================
procedure xuli;
var i,j:longint;
    ok:boolean;
begin
  fillchar(f,sizeof(f),0);

  for i:=1 to k do
    begin
      if (i<a[1]) then begin f[i]:=0; continue; end;

      ok:=false;

      for j:=1 to n do
        begin
          if (i>=a[j]) then
            begin
            if (f[i-a[j]]=1) then continue;
            ok:=true; break;
            end;
        end;

      if ok then f[i]:=1 else f[i]:=0;
    end;

  if f[k]=1 then write('First')
  else write('Second');
end;
//============================
BEGIN
  //assign(input,fi); reset(input);
  //assign(output,fo); rewrite(output);
  doc;
  xuli;
  //close(input); close(output);
END.

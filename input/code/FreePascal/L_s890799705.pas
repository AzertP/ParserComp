{$R-,S-,Q-,I-,O+}
const NN=28;
var a,b,seq:array[0..200009] of int64;
    bits:array[0..NN] of int64;
    n,i,k:longint;
    m,t,ans,bit_count:int64;

procedure quicksort(l,r:longint);
var v,t:int64;i,j:longint;
begin
  if l<r then begin
    v:=seq[r];i:=l-1;j:=r;
    repeat
      repeat inc(i) until v<=seq[i];
      repeat dec(j) until seq[j]<=v;
      t:=seq[i];seq[i]:=seq[j];seq[j]:=t;
    until j<=i;
    seq[j]:=seq[i];seq[i]:=seq[r];seq[r]:=t;
    quicksort(l,i-1);
    quicksort(i+1,r);
  end;
end;

function count(p,q:int64):int64;
var p_idx,q_idx,l,r,m:longint;
begin
  l:=1;r:=n;
  p_idx:=-1;
  while l<=r do begin
    m:=(l+r) div 2;
    if p>seq[m] then
      l:=m+1
    else begin
      p_idx:=m;
      r:=m-1;
    end;
  end;
  if p_idx=-1 then
  begin
    count:=0;
    exit;
  end;

  l:=1;r:=n;
  q_idx:=-1;
  while l<=r do begin
    m:=(l+r) div 2;
    if q>=seq[m] then
      l:=m+1
    else begin
      q_idx:=m;
      r:=m-1;
    end;
  end;
  
  //writeln('DEBUG');
  //writeln('q_idx= ',q_idx);

  if q_idx=-1 then q_idx:=n else dec(q_idx);
  count:=q_idx-p_idx+1;
  
  //write('SEQ ');
  //for i:=1 to n do write(seq[i],',');
  //writeln;
  //writeln(p,'..',q,' = ',count,'--',p_idx,':',q_idx);
  //writeln;
end;

begin
  read(n);
  for i:=1 to n do read(a[i]);
  for i:=1 to n do read(b[i]);
  FillCHar(bits,sizeof(bits),0);
  for k:= 0 to NN do begin
    bit_count:=0;
    m:=1 shl (k+1);
    for i:=1 to n do seq[i]:=b[i] mod m;
    quicksort(1,n);
    for i:=1 to n do begin
      t:=a[i] mod m;
      inc(bit_count,count((1 shl k)-t,(2 shl k)-1-t));
      inc(bit_count,count((1 shl k)-t+(2 shl k),(2 shl k)-1-t+(2 shl k)));
    end;
    bits[k]:=bit_count mod 2;
  end;

  ans:=0;
  for k:=0 to NN do inc(ans,(1 shl k)*bits[k]);
  writeln(ans);
end.

{$R-,S-,Q-,I-,O+}
const KK=1000;
var cond:array[0..1,1..4*KK,1..4*KK] of longint;
    dp:array[0..1,0..4*KK,0..4*KK] of longint;
    n,k,i,x,y,dx,dy,col,row_sum,max,temp:longint;
    c,d:char;

function sum(col,x1,y1,x2,y2:longint):longint;
begin
  sum:=dp[col,x2,y2]+dp[col,x1-1,y1-1]
        -dp[col,x2,y1-1]-dp[col,x1-1,y2];
end;

begin
  readln(n,k);
  for i:=1 to n do begin
    readln(x,y,c,d);
    x:=x mod (2*k);
    y:=y mod (2*k);
    col:=ord(d='W');
    for dx:=0 to 1 do for dy:=0 to 1 do
      inc(cond[col,x+dx*2*k,y+dy*2*k]);
  end;

  for col:=0 to 1 do
    for x:=1 to 4*k do begin
      row_sum:=0;
      for y:=1 to 4*k do begin
        inc(row_sum,cond[col,x,y]);
        dp[col,x,y]:=dp[col,x-1,y]+row_sum;
      end;
    end;

  max:=0;
  for x:=1 to 2*k do for y:=1 to 2*k do begin
    temp:=0;
    inc(temp,sum(0,x+k,y+k,x+2*k-1,y+2*k-1));
    inc(temp,sum(0,x  ,y  ,x+  k-1,y+  k-1));
    inc(temp,sum(1,x+k,y  ,x+2*k-1,y+  k-1));
    inc(temp,sum(1,x  ,y+k,x+  k-1,y+2*k-1));
    if max<temp then max:=temp;
  end;

  writeln(max);
end.

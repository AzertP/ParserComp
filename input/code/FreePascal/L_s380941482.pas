const big_root_priority_queue=true;
      small_root_priority_queue=false;
type priority_queue=object
                      private
                        main:array of longint;
                        size:longint;
                        mode:boolean;
                      public
                        procedure init(a:boolean);
                        procedure push(a:longint);
                        function pop:longint;
                        procedure top;
                        function empty:boolean;
                      end;
procedure priority_queue.init(a:boolean);
begin
  setlength(main,1);
  size:=0;mode:=a;
end;
procedure priority_queue.push(a:longint);
var x,t:longint;
begin
  inc(size);
  setlength(main,size+1);
  main[size]:=a;
  x:=size;
  while x<>1 do
    if (main[x]<main[x>>1]) xor mode then
    begin
      t:=main[x];
      main[x]:=main[x>>1];
      x:=x>>1;
      main[x]:=t;
    end
    else
      break;
end;
function priority_queue.pop:longint;
begin
  exit(main[1]);
end;
procedure priority_queue.top;
var t,x:longint;
begin
  t:=main[1];
  main[1]:=main[size];
  main[size]:=t;
  dec(size);
  setlength(main,size+1);
  x:=1;
  while x<<1<size do
    if ((main[x]<main[x<<1])xor mode)and((main[x]<main[x<<1+1])xor mode) then
      break
    else
    if (main[x<<1]<main[x<<1+1]) xor mode then
    begin
      t:=main[x];
      main[x]:=main[x<<1];
      x:=x<<1;
      main[x]:=t;
    end
    else
    begin
      t:=main[x];
      main[x]:=main[x<<1+1];
      x:=x<<1+1;
      main[x]:=t;
    end;
  if (x<<1=size) and ((main[x]>main[x<<1]) xor mode) then
  begin
    t:=main[x];main[x]:=main[x<<1];main[x<<1]:=t;
  end;
end;
function priority_queue.empty:boolean;
begin
  exit(size<>0);
end;
procedure sort(var a:array of longint;l,r:longint);
var i:longint;
    x:priority_queue;
begin
  x.init(small_root_priority_queue);
  for i:=l to r do
    x.push(a[i]);
  for i:=l to r do
  begin 
    a[i]:=x.pop;
    x.top;
  end;
end;
var n,x,i:longint;
    cnt:array[0..2]of longint;
    a:array[0..100000]of longint;
begin
  readln(n);
  for i:=1 to n do
    read(a[i]);
  sort(a,1,n);
  i:=n;
  while i>=1 do
    if a[i]=a[i-1] then
    begin
      inc(x);
      cnt[x]:=a[i];
      dec(i,2);
      if x=2 then
        break;
    end
    else
      dec(i,1);
  writeln(qword(cnt[1]*cnt[2]));
end.
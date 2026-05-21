program main;
const
  size = 8;

type
  sample = array [1..size] of longint;

var
  n, order : longint;
  p, q : sample;


  procedure input (var a : sample);
  var
    i : longint;

    begin
      for i := 1 to n do
        read(a[i]);
      readln;
    end;

  function code (var a : sample) : sample;
  var 
    i, j, c : longint;
  
    begin
      for i := 1 to n do
        a[i] := a[i] - 1;  //zero index
      for i := 1 to n do begin
        c := 0;
        for j := 1 to (i-1) do
          if (a[j] < a[i]) then inc(c);
        code[i] := a[i] - c;
      end;  //lehmer code
    end;

  function rank (a : sample) : longint;
  var
    i, t : longint;	

      function fact (n : longint) : longint;
        begin
          if (n = 0) or (n = 1) then fact := 1
          else fact := n * fact(n-1);
        end;
  
    begin
      t := 0;
      for i := 1 to n do begin
        t := t + a[i]*fact(n-i);
      end;  //lexicographic rank
	  inc(t);
	  rank := t;
    end;


begin
  readln(n);
  input(p);
  input(q);
  order := abs(rank(code(p))-rank(code(q)));
  writeln(order);
end.


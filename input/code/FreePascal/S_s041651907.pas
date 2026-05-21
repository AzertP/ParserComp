program main;

var N, M, i: integer;
  A: array[1..100] of integer;
  total, count: integer;
  
begin

  readln(N, M);
  for i := 1 to N do read(A[i]);
  total := 0;
  for i := 1 to N do total := total + A[i];
  
  count := 0;
  for i := 1 to N do
    if A[i] >= total/(4*M)
      then count := count + 1;
  
  if count >= M
    then writeln('Yes')
    else writeln('No');
    
end.
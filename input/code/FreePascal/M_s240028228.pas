program main;
type
    arraytype = array[1..200000] of string[10];
var
    i, n, count : longint;
    A : arraytype;

procedure Quicksort(var A : arraytype; left, right : integer);
var
   i, j : integer;
   pivot, temp : string;
begin
    i := left; 
    j := right;
    pivot := A[(left + right) div 2];
    while i <= j do begin
        while A[i] < pivot do 
            i := i + 1;
        while A[j] > pivot do 
            j := j - 1;
        if i <= j then begin 
            temp := A[i];
            A[i] := A[j];
            A[j] := temp;
            i := i + 1;
            j := j - 1;
        end;
    end;
    if left < j
        then Quicksort(A, left, j);
    if i < right 
        then Quicksort(A, i, right);
end;

begin
    //read
    readln(n);
    for i := 1 to n do 
        readln(A[i]);
    
    //sort
    Quicksort(A, 1, n);

    //count
    count := 1;
    for i := 2 to n do
       if A[i] <> A[i-1]
           then count := count + 1;
    writeln(count);
end.
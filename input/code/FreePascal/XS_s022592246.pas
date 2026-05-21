uses math;
var n, k, val : int64;
    
begin
    readln(n, k);
    val := n mod k;
    writeln(min(val, abs(k - val)));
end.
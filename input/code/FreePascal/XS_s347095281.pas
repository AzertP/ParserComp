program main;
var
	num, sum : longint;

begin
	read(num);
    sum := num + (num*num) + (num*num*num);
    writeln(sum);
end.
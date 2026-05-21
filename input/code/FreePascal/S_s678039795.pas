{**
 * Author  : Nguyen Canh Toan
 * Problem : C - Sqrt Inequality
 * Link    : https://atcoder.jp/contests/panasonic2020/tasks/panasonic2020_c
**}
uses math;
const std='tmp';
var a,b,c:int64;
    input,output:text;
procedure main();
var ans:int64;
begin
  read(a,b,c);
  if (c-(a+b+2*sqrt(a)*sqrt(b))>0) then write('Yes') else write('No');
end;
BEGIN
  //assign(input,std+'.inp');reset(input);
  //assign(output,std+'.out');rewrite(output);
    main();
  //close(input);
  //close(output);
END.
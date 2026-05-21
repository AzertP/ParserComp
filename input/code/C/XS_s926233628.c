#include <stdio.h>
int main(){
  int S,h,m,s, r;
  scanf("%d",&S);
  
  h=S/3600;
  r=S%3600;
  m=r/60;
  s=r%60;

printf("%d:%d:%d\n",h,m,s);

return 0;
}
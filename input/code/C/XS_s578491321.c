#include<stdio.h>
int main(){
  int a,b,x;
  scanf("%d%d%d",&a,&b,&x);
  if(a<=x && x<=a+b)
    puts("YES");
  else
    puts("NO");
  return 0;
}

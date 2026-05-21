#include<stdio.h>
#include<math.h>
int main(void){
  int a,b;
  scanf("%d%d",&a,&b);
  printf("%.0f",fmax(a+b,fmax(a-b,a*b)));
  return 0;
}
#include <stdio.h>

int main()
{
  long int a,b,c,d;
  
  scanf("%ld %ld %ld %ld", &a,&b,&c,&d);
  printf("%ld", (a*b>c*d) ? a*b : c*d);
  return 0;
}
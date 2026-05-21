#include <stdio.h>
 
int main (void) {
  int a, b, t;
  int n;
  
  do {
    scanf("%d %d %d", &a, &b, &t);
  } while(!(0<=a && a<=20 && 0<=b && b<=20 && 0<=t && t<=20)); {
    scanf("%d %d %d", &a, &b, &t);
  }
  n = t / a;
  printf("%d\n", b * n);
  
  return 0;
}
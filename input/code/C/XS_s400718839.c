#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
  double r;
  scanf("%lf",&r);
  printf("%.6f %.6f\n", r*r*M_PI, 2.0*r*M_PI);
  return 0;

}

#include <stdio.h>

int main(void) {
  int N;
  int V[20];
  int C[20];
  scanf("%d", &N);
  for(int i = 0; i < N; i++) {
    scanf("%d", &V[i]);
  }
  for(int i = 0; i < N; i++) {
    scanf("%d", &C[i]);
  }

  int X = 0;
  for(int i = 0; i < N; i++) {
    if(V[i] > C[i])
      X += (V[i] - C[i]);
  }
  
  printf("%d\n", X);
  return 0;
}

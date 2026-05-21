#include<stdio.h>

int Order(int *, int);
int T[101][101];

int main(){
  int n, i;
  int p[101];

  scanf("%d", &n);

  for(i = 0; i < n; i++){
    scanf("%d%d", &p[i], &p[i+1]);
  }

  printf("%d\n", Order(p, n));

  return 0;
}


int Order(int *p, int n){
  int i, j, k, l, q;

  for(i = 1; i <= n; i++){
    T[i][i] = 0;
  }

  for(l = 2; l <= n; l++){
    for(i = 1; i <= n - l + 1; i++){
      j = i + l - 1;

      T[i][j] = 20000000;

      for(k = i; k <= j - 1; k++){
        q = T[i][k] + T[k + 1][j] + p[i - 1] * p[k] * p[j];
        T[i][j] = ((T[i][j] <= q) ? T[i][j] : q);
      }

    }
  }

  return T[1][n];
}
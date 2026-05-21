#include <stdio.h>

int main (void) {
  int n, i, ans;
  int p[20];

  scanf ("%d", &n);

  for (i = 0; i < n; i++) {
    scanf ("%d", &p[i]);
  }

  for (i = 1; i < n-1; i++) {
    if (p[i] > p[i-1] && p[i] < p[i+1]) {
      ans++;
    } else if (p[i] < p[i-1] && p[i] > p[i+1]) {
      ans++;
    }
  }

  printf ("%d", ans);
  
}
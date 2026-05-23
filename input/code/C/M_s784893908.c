
int main(void)
{
  int n = 0;
  int m = 0;
  int a[101][101];
  int i, j;
  int sum = 0;
  
  scanf("%d %d", &n, &m);

  for(i = 0; i < n; i++){
    for(j = 0; j < m; j++)
      scanf("%d ", &a[i][j]);
  }


  for(i = 0; i < n; i++){
    for(j = 0; j < m; j++)
      sum += a[i][j];
    a[i][j] = sum;
    sum = 0;
  }

  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++)
      sum += a[j][i];
    a[j][i] = sum;
    sum = 0;
  }

  for(i = 0; i < m; i++)
      sum += a[n][i];
  a[n][m] = sum;
  

  for(i = 0; i <= n; i++) {
    for(j = 0; j < m; j++) 
      printf("%d ", a[i][j]);
    
    printf("%d\n", a[i][j]);
  }
  
  return 0;
}

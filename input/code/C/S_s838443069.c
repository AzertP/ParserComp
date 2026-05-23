int main()
{
  int x[N];
  int i;
  int max = -1000000000;
  int sa;
  int min;

  for(i = 0 ; i < N ; i++){
    scanf("%d", &x[i]);
  }

  min = x[1];

  for(i = 2 ; i < x[0] + 1 ; i++){
    sa = x[i] - min;
    if(max < sa){
      max = sa;
    }
    if(min > x[i]){
      min = x[i];
    }
  }

  printf("%d\n", max);
  return 0;
}

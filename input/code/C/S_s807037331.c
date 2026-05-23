

int F[MAX];

int fibonacci(int);

int main()
{
  int n, i;
  int fib[MAX];

  scanf("%d", &n);

  for( i = 0 ; i < MAX ;i++ ) F[i] = -1;

  printf("%d\n", fibonacci(n));
  
  return 0;
}

int fibonacci(int n)
{
  if( n == 0 || n == 1 ) return F[n] = 1;
  if( F[n] != -1 ) return F[n];
  
  return F[n] = fibonacci(n - 2) + fibonacci(n - 1);
}


int main(void)
{
  int a, b, c, min;
  
  scanf("%d%d%d", &a, &b, &c);

  min = a+b;
  if (a+c<min) min = a+c;
  if (b+c<min) min = b+c;
  
  printf("%d\n", min);
  
  return 0;
}

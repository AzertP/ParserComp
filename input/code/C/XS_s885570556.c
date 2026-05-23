
int gcd(int a,int b){
  if(a==b) return a;
  else if(a>b) return gcd(a-b,b);
  else return gcd(a,b-a);
  return 0;
}

int main(void){
  int x,y;
  scanf("%d %d",&x,&y);
  printf("%d\n",gcd(x,y));
  return 0;
}
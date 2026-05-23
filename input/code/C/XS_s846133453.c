int main(void){
  int n,x=0,i,xmax=0;
  char s[101];
  scanf("%d",&n);
  scanf("%s",s);
  for(i=0;i<n;i++){
    if(s[i]=='I')x++;
    else if(s[i]=='D')x--;
    if(x>xmax)xmax=x;
  }
  printf("%d",xmax);
  return 0;
}
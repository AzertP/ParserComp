int main(){
  int n,t,i,x,y,a=(1<<29)*-1,b=1<<29;
  scanf("%d",&n);
  for(i=1;i<=n;i++){scanf("%d",&t);if(t>a){x=i;a=t;}if(t<b){y=i;b=t;}}
  printf("%d\n",n*2-1);
  if(-b>a){for(i=1;i<=n;i++)printf("%d %d\n",y,i);for(i=n;i>1;i--)printf("%d %d\n",i,i-1);return 0;}
  for(i=1;i<=n;i++)printf("%d %d\n",x,i);for(i=1;i<n;i++)printf("%d %d\n",i,i+1);
}

int min(int,int);
int main(){
  int i,j,k,l,n,p[N+1],m[N+1][N+1];
  scanf("%d",&n);
  for(i = 1;i <= n;i++){
    scanf("%d%d",&p[i-1],&p[i]);
  }
  for(i = 1;i <= n;i++){
    m[i][i]=0;
  }
  for(l = 2;l <= n;l++){
    for(i = 1;i <= n-l+1;i++){
      j=i+l-1;
      m[i][j]=INF;
      for(k = i;k < j;k++){
        m[i][j]=min(m[i][j],m[i][k]+m[k+1][j]+p[i-1]*p[k]*p[j]);
      }
    }
  }
  printf("%d\n",m[1][n]);
  return 0;
}
int min(int x,int y){
  if(x > y){
    return y;
  }
  else return x;
}

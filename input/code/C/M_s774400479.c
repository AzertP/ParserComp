 
 
 
int p[100],m[100][100];
void MCM(int n){
  int i,l,j,k,b;
  for(l=2;l<=n;l++){
    for(i=1;i<=n-l+1;i++){
      j = i + l - 1;
      m[i][j] = 100000;
      for(k=i;k<j;k++){
        b = m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j];
        if(m[i][j]>b) m[i][j]=b;
      }
    }
  }
}
 
int main(){
  int n,i,j;
 
  scanf("%d",&n);
  for(i=1;i<=n;i++){
    scanf("%d%d",&p[i-1],&p[i]);
  }
 
  for(i=1;i<=n;i++){
    m[i][i]=0;
  }
  MCM(n);
  printf("%d\n",m[1][n]);
 
  return 0;
}

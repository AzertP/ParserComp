 
int main(void){
  int m,n,i,j,tmp;
  int a[100];
  int all_vote=0;
  
  do{
    scanf("%d %d",&n,&m);
  }while(m<=0 || n<m || 101<=n);
  
  for(i=0; i<n; i++){
    do{
      scanf("%d",&a[i]);
    }while(a[i]<1 || a[i]>1000);
    all_vote+=a[i];
  }
  
  for(i=0; i<n; i++){
    for(j=i+1; j<n; j++){
      if(a[i] < a[j]){
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
    }
  }
  
  if(4*m*a[m-1] >= all_vote){
    printf("Yes");
  }
  else
    printf("No");
  
  return 0;
}

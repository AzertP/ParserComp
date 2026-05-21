#include<stdio.h>
int main(){
  int i,j,k,N,count=0,flag,v;
  int data[100];
  scanf("%d",&N);
  
  for(i=0;i<N;i++){
    scanf("%d",&data[i]);
  }
  flag=1;
  while(flag){
    flag=0;
    for(j=N-1;j>=1;j--){
      if(data[j]<data[j-1]){
	v=data[j];
	data[j]=data[j-1];
	data[j-1]=v;
	flag=1;
	count++;
      }
    }
  }
  for(k=0;k<N;k++){
    printf("%d",data[k]);
    if(k!=N-1)printf(" ");
  }
  printf("\n");
  
  printf("%d\n",count);
  return 0;
}


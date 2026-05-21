#include<stdio.h>
int snk(int);
int main(void){
  int N,A,sum=1000;
  scanf("%d",&N);
  for(int i=0;i<N;i++){
    scanf("%d",&A);
    if(snk(A)<sum){
      sum=snk(A);
    }
  }
  if(sum!=1000){
    printf("%d\n",sum);
  }else{
    printf("0\n");
  }
  return 0;
}
int snk(int X){
  int j=0;
  while(1){
    if(X%2==0){
      X/=2;
      j++;
    }else{
      break;
    }
  }
  return j;
}
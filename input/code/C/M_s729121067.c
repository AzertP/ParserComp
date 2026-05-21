#include<stdio.h>
int selectionSort(int A[],int N);

int main(){
  int A[100];
   int N,i,result;
   
 scanf("%d",&N);
 for(i=0;i<N;i++){
   scanf("%d",&A[i]);
 }
 
   result=selectionSort(A,N);
 
 for(i=0;i<N;i++){
   
  if(i>0)printf(" ");
  printf("%d",A[i]);
  
 }
 printf("\n");
 printf("%d\n",result);
 
 return 0;
}

int selectionSort(int A[],int N){
  int i,j;
  int tench,result=0;
  int minj;
 
 for(i=0;i<N-1;i++){
   
  minj=i;
  
  for(j=i;j<N;j++){
    
    if(A[j]<A[minj]){
      
     minj=j;
    }
    
  }
  tench=A[i];
  
  A[i]=A[minj];
  A[minj]=tench;
  
  if(i!=minj){
    result++; }
 }
 return result;
}


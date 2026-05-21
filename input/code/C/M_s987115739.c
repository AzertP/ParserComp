#include <stdio.h>
#include <stdlib.h>

int n;

int Parent(int a){
  return a/2;
}

int Left (int a){
  return a*2;
}
int Right (int a){
  return a*2 + 1;
}

void swap (int *A,int a,int b){
  int cmp;
 
  cmp = A[a];
  A[a] = A[b];
  A[b] = cmp;
  
}

void maxheapify(int *A,int i){
  int l,r,largest;

  l = Left(i);
  r = Right(i);

  if(l <= n && A[l] > A[i]) largest = l;
  else largest = i;

  if(r <= n && A[r] > A[largest]) largest = r;

  if(largest != i) {
    swap(A,i,largest);
    maxheapify(A,largest);
  }

}

void  buildMaxHeap(int *A){
  int i;

  for(i=n/2;i>0;--i) maxheapify(A,i);
}

int main (){
  int i;
  int *A;

  scanf("%d",&n);

  A = malloc(sizeof(int) * (n+2));

  for(i=1;i<=n;++i){
    scanf("%d",&A[i]);
  }

  buildMaxHeap(A);

  for(i=1;i<=n;++i) printf(" %d",A[i]);
  printf("\n");

  return 0;
}


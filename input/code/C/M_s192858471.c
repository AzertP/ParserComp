#include<stdio.h>
#include<stdlib.h>

int h;

int p(int i);
int l(int i);
int r(int i);
void swap(int *a, int *b);
void maxHeapify(int *A, int i);
void buildMaxHeap(int *A);

int main(){
  int i,*A;
  scanf("%d",&h);
  A=malloc(sizeof(int)*(h+1));
  for(i=1;i<=h;i++){
    scanf("%d",&A[i]);
  }
  buildMaxHeap(A);
  for(i=1;i<=h;i++){
    printf(" %d",A[i]);
  }
  printf("\n");
  return 0;
}

int p(int i){
  return i/2;
}

int l(int i){
  return i*2;
}

int r(int i){
  return i*2+1;
}

void swap(int *x,int *y){
  int temp;
  temp=*x;
  *x=*y;
  *y=temp;
}

void maxHeapify(int *A, int i){
  int left,right,largest;
  left=l(i);
  right=r(i);
  if(left<=h&&A[left]>A[i]){
    largest=left;
  }
  else{
    largest=i;
  }
  if(right<=h&&A[right]>A[largest]){
    largest=right;
  }
  if(largest!=i){
    swap(&A[i],&A[largest]);
    maxHeapify(A,largest);
  }
}

void buildMaxHeap(int *A){
  int i;
  for(i=h/2;i>0;i--){
    maxHeapify(A,i);
  }
}


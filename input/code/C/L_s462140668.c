#include<stdio.h>
#include<math.h>

#define N 100000

typedef struct{
  char mark;
  int num;
} Trump;

void merge(Trump*,int,int,int);
void mergeSort(Trump*,int,int);
int partition(Trump*,int,int);
void quickSort(Trump*,int,int);

Trump A[N];

int main(void){
  int n;
  Trump B[N],C[N];
  int i;

  scanf(" %d",&n);
  for(i = 0; i < n; i++){
    scanf(" %c %d",&A[i].mark,&A[i].num);
    C[i].mark = B[i].mark = A[i].mark;
    C[i].num = B[i].num = A[i].num;
  }

  mergeSort(C,0,n);

  quickSort(B,0,n-1);

  for(i = 0; i < n; i++){
    if(C[i].mark != B[i].mark){
      printf("Not stable\n");
      break;
    }
  }
  if(i == n)printf("Stable\n");

  for(i = 0; i < n; i++){
    printf("%c %d\n",B[i].mark,B[i].num);
  }

  
  return 0;
}

void quickSort(Trump *B, int p, int r){
  int q;
  if(p < r){
    q = partition(B,p,r);
    quickSort(B,p,q-1);
    quickSort(B,q+1,r);
  }
}

int partition(Trump *B, int p, int r){
  int x=B[r].num;
  int i=p-1,j;
  Trump tmp;
  for(j = p; j < r; j++){
    if(B[j].num <= x){
      i++;
      tmp = B[i];
      B[i] = B[j];
      B[j] = tmp;
    }
  }
  tmp = B[i+1];
  B[i+1] = B[r];
  B[r] = tmp;
  return i+1;
}

void merge(Trump *C, int left, int mid, int right){
  int n1,n2;
  int i,j,k;
  Trump L[N],R[N];
  n1 = mid - left;
  n2 = right -mid;
  for(i = 0; i < n1; i++){
    L[i] = C[left + i];
  }
  for(i = 0; i < n2; i++){
    R[i] = C[mid + i];
  }
  L[n1].num = (int)INFINITY;
  R[n2].num = (int)INFINITY;
  i = 0;
  j = 0;
  for(k = left; k < right; k++){
    if(L[i].num <= R[j].num){
      C[k] = L[i];
      i++;
    }
    else{
      C[k] = R[j];
      j++;
    }
  }
}

void mergeSort(Trump *C, int left, int right){
  int mid;
  if(left+1 < right){
    mid = (left + right)/2;
    mergeSort(C,left,mid);
    mergeSort(C,mid,right);
    merge(C,left,mid,right);
  }
}


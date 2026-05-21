#include <stdio.h>
#include <stdlib.h>

#define D_MAX 1000000000
#define N_MAX 100000
#define M_MAX 10000
#define min(x,y) (((x)<(y))?(x):(y))

int d;
int n;
int m;

int ad[N_MAX+1];
int ak[M_MAX+1];

void MergeSort(int array[],int left,int right){
  int awork[N_MAX+1];
  int mid;
  int i;
  int j;
  int x;
  int y;
  
  if( left >= right ){
    return;
  }
  
  mid = (left + right) / 2;
  MergeSort( array, left, mid);
  MergeSort( array, mid+1, right);
  
  for( i=left; i<=mid; i++){
    awork[i] = array[i];
  }
  
  j = right;
  for( i=mid+1; i<=right; i++){
    awork[i] = array[j];
    j--;
  }
  
  x = left;
  y = right;
  
  for( i=left; i<=right; i++){
    if(awork[x] <= awork[y]){
      array[i] = awork[x];
      x++;
    }else{
      array[i] = awork[y];
      y--;
    }
  }
}


int calcDist(void){
  int i;
  int j = 1;
  int sum = 0;;
  int minDist = d;
  
  for( i=1; i<=m; i++){
    while(ad[j] < ak[i]){
      j++;
    }
    minDist = min( ak[i]-ad[j-1],ad[j]-ak[i] );
    sum += minDist;
  }
  return sum;
}


void input(void){
  int i;
  scanf("%d",&d);
  if( d != 0){
    scanf("%d",&n);
    scanf("%d",&m);
    ad[0] = 0;
    for( i=1; i<=n-1; i++){
      scanf("%d",&ad[i]);
    }
    ad[n] = d;
    for( i=1; i<=m; i++){
      scanf("%d",&ak[i]);
    }
  }
}

int main(void){
  int i;
  int sumDist = 0;
  while( 1 ){
    input();
    if( d == 0 ){
      break;
    }
    MergeSort( ad, 0, n);
    MergeSort( ak, 1, m);
    sumDist = calcDist();
    printf("%d\n",sumDist);
    sumDist = 0;
  }
}



#include<stdio.h>
#include<stdlib.h>
#define INF (1e9+1)
 
typedef struct state{
  int pos,cost;
} state;
 
state A[2000001];
int N=0;
  
int parent(int i){return i/2;}
int left(int i){return i*2;}
int right(int i){return i*2+1;}
  
void exchange(state* a,state* b){
  state tmp=*a;
  *a=*b;
  *b=tmp;
}
  
void minHeapify(int i){
  int l=left(i);
  int r=right(i);
  int small;
   
  if( l<=N && A[l].cost<A[i].cost )small=l;
  else small=i;
   
  if( r<=N && A[r].cost<A[small].cost)small=r;
   
  if(small!=i){
    exchange(&A[i],&A[small]);
    minHeapify(small);
  }
   
}
  
state heapExtractMin(){
  if(N<1)return (state){0,0};
  state min=A[1];
  A[1]=A[N];
  N--;
  minHeapify(1);
  return min;
}
 
  
void heapIncreaseKey(int i,state key){
  if(key.cost>A[i].cost)return;
  
  A[i]=key;
  while(i>1&&A[parent(i)].cost>A[i].cost){
    exchange(&A[i],&A[parent(i)]);
    i=parent(i);
  }
}
  
void minHeapInsert(state key){
  N++;
  A[N]=(state){INF,INF};
  heapIncreaseKey(N,key);
}
 
 
typedef struct edge{
  int to,cost;
} edge;
 
edge* G[10000];
int size[10000];
int n;
int d[10000];
edge e;
state s;
int main(){
  int i,j,a,b;
  scanf("%d",&n);
  for(i=0;i<n;i++){
    scanf("%d %d",&a,&b);
    G[a] = malloc( sizeof( edge ) *b );
    size[a]=b;
    for(j=0;j<b;j++){
      scanf("%d %d",&G[a][j].to,&G[a][j].cost);
    }
    d[i]=INF;
  }
  d[0]=0;
  minHeapInsert( (state){0,0} );
  while(N!=0){
    s = heapExtractMin();
    if(s.cost>d[s.pos])continue;
    for(i=0;i<size[s.pos];i++){
      e=G[s.pos][i];
      if(d[e.to]<=d[s.pos]+e.cost)continue;
      d[e.to]=d[s.pos]+e.cost;
      minHeapInsert((state){e.to,d[e.to]});
    }
  }
  for(i=0;i<n;i++)
    printf("%d %d\n",i,d[i]);
   
  return 0;
}


int parent(int);
int left(int);
int right(int);
void maxHeapify(int *,int);
void buildMaxHeap(int *);


int heap[H],n;

int main(){

  int i,j;

  scanf("%d",&n);

  for(i=1;i<=n;i++) scanf("%d",&heap[i]);

  buildMaxHeap(heap);

  for(j=1;j<=n;j++) printf(" %d",heap[j]);
  printf("\n");

  return 0;

}

int parent(int x){
  return (x/2);
}

int left(int x){
  return x*2;
}

int right(int x){
  return (x*2+1);
}

void maxHeapify(int h[],int x){

  int l,r;

  int largest,tmp;

  l=left(x);
  r=right(x);

  if(l<=n && h[l]>h[x]){
    largest=l;
  }
  else{
    largest=x;
  }

  if(r<=n && h[r]>h[largest]){
    largest=r;
  }

  if(largest!=x){

    tmp=h[x];
    h[x]=h[largest];
    h[largest]=tmp;
    // printf("%d\n",x);
    maxHeapify(h, largest);
  }

}

void buildMaxHeap(int h[]){

  int k;

  for(k=n/2;k!=0;k--) maxHeapify(h,k);
}

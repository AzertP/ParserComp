
int main() {

  int N=0,M=0,A=0,B=0;
  int H[100000]={0};
  bool great[100000]={false};
  int count=0;
  scanf("%d %d",&N,&M);
  for(int i=0;i<N;i++){
    great[i]=true;
  }
  for(int i=0;i<N;i++) scanf("%d",&H[i]);
  for(int i=0;i<M;i++){
    scanf("%d %d",&A,&B);
    if(H[A-1]<=H[B-1]){
      great[A-1]=false;
    }
    if(H[A-1]>=H[B-1]){
      great[B-1]=false;
    }
  }
  for(int i=0;i<N;i++){
    if(great[i]) {
      count++;
    }
  }
  printf("%d\n",count);
  return 0;
}

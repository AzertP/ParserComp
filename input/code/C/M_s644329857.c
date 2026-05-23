void f_post(int *,int *,int,int,int,int*);
void post(int *,int *,int,int,int,int*);
int main(){
  int i,n,count=0;
  int *pre,*in;

  scanf("%d",&n);
  pre = malloc(sizeof(int)*n);
  in = malloc(sizeof(int)*n);
  for(i=0;i<n;i++) scanf("%d",&pre[i]);
  for(i=0;i<n;i++) scanf("%d",&in[i]);
  f_post(pre,in,0,n,n,&count);
  free(pre);
  free(in);
  return 0;
}
void f_post(int *pre,int *in,int l,int r,int n,int *count){
  int i,j,m,num;
  if(l>=r)return;
  num = pre[(*count)++];
  for(i=0;i<n;i++){
    if(num == in[i])break;
  }
  post(pre,in,l,i,n,count);
  post(pre,in,i+1,r,n,count);
  printf("%d\n",num);
}
void post(int *pre,int *in,int l,int r,int n,int *count){
  int i,j,m,num;
  if(l>=r)return;
  num = pre[(*count)++];
  for(i=0;i<n;i++){
    if(num == in[i])break;
  }
  post(pre,in,l,i,n,count);
  post(pre,in,i+1,r,n,count);
  printf("%d ",num);
}
       
  
  


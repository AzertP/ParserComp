#include<stdio.h>
#include<stdlib.h>

int compare_int(const void *a, const void *b)
{
  return *(int*)a-*(int*)b;
}

int main()
{
  int n; 
  int a[100001];
  int i;
  int ans=0;
  int s=0,t=0;
  scanf("%d",&n);
  for(i=0;i<n;i++){
    scanf("%d",&a[i]);
  }
  qsort(a,n,sizeof(int),compare_int);
  for(i=0;i<n;i++){
    if(s==a[i]){
      if(t==0){
        t=1;
        ans++;
      }
      else if(t==1){
        ans--;
        t=0;
      }
    }
    else if(s!=a[i]){
      s=a[i];
      t=1;
      ans++;
    }
  }
  printf("%d",ans);
  return 0;
}

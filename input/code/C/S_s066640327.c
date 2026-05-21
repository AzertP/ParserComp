#include<stdio.h>
#define Max 10000

int main(){
  int i,j,count,n,q,S[Max],T[Max];
 
  scanf("%d",&n);
  for(i=0;i<n;i++){
    scanf("%d",&S[i]);
  }
  scanf("%d",&q);
  for(i=0;i<q;i++){
    scanf("%d",&T[i]);
  }

  count=0;

   for(i=0;i<q;i++){
      S[n]=T[i];
      for(j=0;j<=n;j++){
      if(j==n) break;
      else if(S[j]==T[i]){
        count++;
        break;
      }
    }
    }
  printf("%d\n",count);
  return 0;
}
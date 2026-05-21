#include <stdio.h>

int main(void) {
  int n,i,a[100],ap=1;
  scanf("%d",&n);
  for(i=0;i<n;i++){
    scanf("%d",&a[i]);
    if(a[i]%2==0){
      if(a[i]%3!=0&&a[i]%5!=0){
        ap=0;
      }
    }
  }
  if(ap==0){
    printf("DENIED");
  }
  else printf("APPROVED");
}
#include<stdio.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>
#include<stdlib.h>
#define rep(i,n) for(int i=0; i <(n); i++)
int main(){
  int n;
  long a[100000];
  scanf("%d",&n);
  rep(i,n){
    scanf("%lld",&a[i]);
  }
  int x=0,y=0,z=0;
  rep(j,n){
    if(a[j]%4==0){
      x++;
    }else if(a[j]%2==0){
      y++;
    }else{
      z++;
    }
  }
  if(x==z-1){
    if(n%2==1&&y==0){
      printf("Yes\n");
      return 0;
    }
  }else if(x>z-1){
    printf("Yes\n");
    return 0;
  }
  printf("No\n");
  return 0;
}

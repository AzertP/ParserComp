#include<stdio.h>
int main(){
	int n;
  scanf("%d",&n);
  int sum=0;
  int m=n;
  while(m!=0){
   sum+=m%10;
    m/=10;
  }
  if(n%sum==0){
    puts("Yes");
  }else{
      puts("No");
  }
  
  return 0;
}
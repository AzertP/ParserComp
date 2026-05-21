#include <stdio.h>
int main(){
    int X, Y;
    
    scanf("%d %d",&X,&Y);
    if(Y%2==1){
      printf("No");
    }else if(2*X<=Y && Y<=4*X){
      printf("Yes");
    }else{
      printf("No");
    }
return 0;
}
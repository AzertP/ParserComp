#include <stdio.h>

int main(void){
  int H,W,N;
  scanf("%d%d%d",&H,&W,&N);

  int a,counter = 0;
  int data[H*W];
  int ans[H][W];
  for(int i = 0;i < N;i++){
    scanf("%d",&a);
    int num = a;
    while(num > 0){
      data[counter+num-1] = i+1;
      num--;
    }
    counter += a;
  }

  counter = 0;
  for(int i = 0;i < H;i++){
    if(i%2 == 0){
      for(int j = 0;j < W;j++){
        ans[i][j] = data[counter];
        counter++;
      }
    }else{
      for(int j = W-1;j >= 0;j--){
        ans[i][j] = data[counter];
        counter++;
      }
    }
  }

  for(int i = 0;i < H;i++){
    for(int j = 0;j < W;j++){
      printf("%d",ans[i][j]);
      if(j != W-1) printf(" ");
    }
    printf("\n");
  }


  return 0;
}
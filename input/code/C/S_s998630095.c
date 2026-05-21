#include <stdio.h>

int main(){
  int num,yen;
  scanf("%d %d", &num, &yen);
  yen /= 1000;
  int sum,k;
  int answer[3] = {-1,-1,-1};
  for(int i = 0; i <= num; ++i){
    if(10 * i > yen) break;
    for(int j = 0; j <= num - i; ++j){
      k = num - i - j;
      sum = 10 * i + 5 * j + k;
      if(sum == yen){
        answer[0] = i;
        answer[1] = j;
        answer[2] = k;
      }else if(sum > yen) break;
    }
  }
  printf("%d %d %d\n",answer[0],answer[1],answer[2]);
  return 0;
}
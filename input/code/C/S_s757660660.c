#include <stdio.h>

int main(void){
    int N,i,j,count;
    while(scanf("%d",&N),N){
        j=1;
        count=0;
        for(i=2;j<=N;i++){
            j+=i;
            if((N-j)%i==0&&N-j>=0){
                count++;
                //printf("i:%d\n",i);
            }
        }
        printf("%d\n",count);
    }
    return 0;
}
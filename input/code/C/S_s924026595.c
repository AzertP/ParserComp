#include <stdio.h>

int main(int argc,const char * argv[]){
    int i,j,H,W;
    
    for(;;){
        scanf("%d %d",&H,&W);
        if(H==0 && W==0)break;
        for(i=1;i<=H;i++){
            for(j=1;j<=W;j++){
                if((i%2==0&&j%2==0)||(i%2==1&&j%2==1)){
                    printf("#");
                }else{
                    printf(".");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    return 0;
}
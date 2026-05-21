#include <stdio.h>

int main(int argc, const char * argv[]) {
    int r,c;
    int i,j;
    scanf("%d %d",&r,&c);//r:???c:?¨????r???c???
    int array[101][101]={0};
    for(j=0;j<r;j++){//????????????
        for(i=0;i<c;i++){//?¨??????????
            scanf("%d",&array[i][j]);
            array[c][j]+=array[i][j];
            array[i][r]+=array[i][j];
            array[c][r]+=array[i][j];
            
        }
    }for(j=0;j<=r;j++){
        if(j!=0){
            printf("\n");
        }
        for(i=0;i<=c;i++){
            if(i!=0){
                printf(" ");
            }
            printf("%d",array[i][j]);
                    }
    }printf("\n");
    return 0;
}
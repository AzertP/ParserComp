#include<stdio.h>

int main(){
    int x=0,y=0,z=0;
    for(x=1;x<=9;x++){
        for(y=1;y<=9;y++){
        printf("%dx%d=%d\n",x,y,x*y);
        }
    }
    return 0;
}


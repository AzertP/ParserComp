#include <stdio.h>
int main()
{
    int a,b;
    scanf("%d %d",&a,&b);
    if(a+b==15){
        printf("+\n");
    }
    else if(a*b==15){
        printf("*\n");
    }
    else printf("x\n");


    return 0;
}

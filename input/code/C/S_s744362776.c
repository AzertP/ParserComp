#include <stdio.h>

int main(void)
{
    int a,i,j,k,sum;
    sum=0;

    scanf("%d%d%d",&i,&j,&k);
    for(a=i;a<=j;a++)
    {
        if((k%a)==0)
        sum+=1;
    }
    printf("%d\n",sum);

    return 0;
}
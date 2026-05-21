#include<stdio.h>
int main(void)
{
    int a,b;
    scanf("%d",&a);   
    scanf("%d",&b);
    int x=(a+b)/2;
    if((a+b)%2==0)
    {printf("%d\n",x);
    }
    else
    {x=x+1;
    printf("%d\n",x);
    }
    return 0;
}
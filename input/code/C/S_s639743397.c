#include<stdio.h>
int main()
{
    int a,b,t,n;
    scanf("%d %d",&a,&b);

    if(b>a)
    {
        t=b;
        b=a;
        a=t;
    }
    n=a/b;
    if(n>=2)
    {
        printf(":(");
    }
    else
    {
        printf("Yay!");
    }
    return 0;

}

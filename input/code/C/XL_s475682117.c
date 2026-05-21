#include<stdio.h>
int main(void)
{
    int a,b,c,d,e,box;
    while(scanf("%d,%d,%d,%d,%d",&a,&b,&c,&d,&e)!=EOF)
    {
         if(a>b)
        {
            box=a;
            a=b;
            b=box;
        }
        if(b>c)
        {
            box=b;
            b=c;
            c=box;
        }
        if(c>d)
        {
            box=c;
            c=d;
            d=box;
        }
        if(d>e)
        {
            box=d;
            d=e;
            e=box;
        }
        if(a>b)
        {
            box=a;
            a=b;
            b=box;
        }
        if(b>c)
        {
            box=b;
            b=c;
            c=box;
        }
        if(c>d)
        {
            box=c;
            c=d;
            d=box;
        }
        if(a>b)
        {
            box=a;
            a=b;
            b=box;
        }
        if(b>c)
        {
            box=b;
            b=c;
            c=box;
        }
        if(a>b)
        {
            box=a;
            a=b;
            b=box;
        }
        if(a==b)
        {
            if(b==c)
            {
                if(c==d||c==e)
                    printf("four card\n");
                else if(d==e)
                    printf("full house\n");
                else
                    printf("three card\n");
                    }
            else if(c==d)
            {
                if(a==e||c==e)
                    printf("full house\n");
                else
                    printf("two pair\n");
            }
            else if(d==e)
            {
                if(a==d)
                    printf("four card\n");
                else
                    printf("two pair\n");
            }
            else if(b==d)
                printf("three card\n");
            else if(b==e)
            {
                if(c==d)
                    printf("full house\n");
                else
                    printf("three card\n");
            }
            else if(c==d)
                printf("two pair\n");
            else if(c==e)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(a==c)
        {
            if(c==d)
            {
                if(d==e)
                    printf("four card\n");
                else if(b==e)
                    printf("full house\n");
                else
                    printf("three card\n");
            }
            else if(c==e)
            {
                if(b==d)
                    printf("full house\n");
                else
                    printf("three card\n");
            }
            else if(b==d)
                printf("two pair\n");
            else if(b==e)
                printf("two pair\n");
            else if(d==e)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(a==d)
        {
            if(a==e)
            {
                if(b==c)
                    printf("full house\n");
                else
                    printf("three pair\n");
            }
            else if(b==c)
            {
                if(b==e)
                    printf("full house\n");
                else
                    printf("two pair\n");
            }
            else if(b==e||c==e)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(a==e)
        {
            if(b==c)
            {
                if(c==d)
                    printf("full house\n");
                else
                    printf("two pair\n");
            }
            else if(b==d||c==d)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(b==c)
        {
            if(c==d)
            {
                if(d==e)
                    printf("four card\n");
                else
                    printf("three card\n");
            }
            else if(c==e)
                printf("three card\n");
            else if(d==e)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(b==d)
        {
            if(d==e)
                printf("three card\n");
            else if(c==e)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(b==e)
        {
            if(c==d)
                printf("two pair\n");
            else
                printf("one pair\n");
        }
        else if(c==d)
        {
            if(d==e)
                printf("three card\n");
            else
                printf("one pair\n");
        }
        else if(c==e)
            printf("one pair\n");
        else if(d==e)
            printf("one pair\n");
        else if(e==d+1&&d==c+1&&c==b+1&&b==a+1)
            printf("straight\n");
        else if(e==d+1&&d==c+1&&c==b+1&&b==a+9)
            printf("straight\n");
        else printf("null\n");
    }
    return 0;
}
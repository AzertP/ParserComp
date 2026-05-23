int max(const void *a, const void *b){return *(int*)b - *(int*)a;}


int main(void)
{
    int n;
    scanf("%d",&n);

    if(n == 100000)
    {
        printf("%d\n", 9+ 900+ 90000);
    }else if( n >= 10000)
    {
        printf("%d\n", 9+ 900+ (n-10000+1));
    }else if( n >= 1000)
    {
        printf("%d\n", 9+ 900);
    }else if( n >= 100)
    {
        printf("%d\n", 9+ (n-100+1));
    }else if( n >= 10)
    {
        printf("%d\n",9);
    }else 
    {
        printf("%d\n", n);
    }

    
    return 0;
}

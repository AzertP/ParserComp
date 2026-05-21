#include <stdio.h>

int main(void)
{
    int N;
    scanf("%d", &N);
    int i;
    int a[100000];
    for(i = 1; i <= N; i++)
        scanf("%d", &a[i]);
    int now_count = 1;
    int count;
    for(count = 1;count <= N; count++)
    {
        //printf("Count: %d Now Count: %d\n",count,now_count);
        if(a[now_count] == 2)
        {
            printf("%d\n",count);
            return 0;
        }
        else
            now_count = a[now_count];
        //printf("Count: %d Now Count: %d\n",count,now_count);
    }
    printf("-1\n");
    return 0;
}
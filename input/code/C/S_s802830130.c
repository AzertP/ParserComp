#include<stdio.h>
#include<string.h>
int main()
{
    int a,b,i,j,n;
    scanf("%d",&n);
    char ch[10000],c,m;
    c=getchar();
    gets(ch);
    for(i=0;ch[i]!='\0';i++)
    {
        ch[i]=ch[i]+n;
        if(ch[i]>'Z')
        {
            ch[i]=ch[i]-'Z';
            ch[i]='A'+ch[i]-1;
        }
    }
    printf("%s\n",ch);
    return 0;
}

#include <stdio.h>

int cal(int a[],int b[],char str[][1],int i)
{
        int s;

        if (str[i][0]=='+') { s=a[i]+b[i]; }
        else if (str[i][0]=='-') { s=a[i]-b[i]; }
        else if (str[i][0]=='*') { s=a[i]*b[i]; }
        else if (str[i][0]=='/') { s=a[i]/b[i]; }

        return s;
}

int main(void)
{
        int a[100],b[100];
        char str[100][1];
        int ans[100];
        int i,j;

        i=0;
        while(1) {
                scanf("%d",&a[i]);
                scanf("%s",str[i]);
                scanf("%d",&b[i]);
          if (str[i][0]=='?') { break; }
                ans[i]=cal(a,b,str,i);
          i++;
        }

  for (j=0; j<i; j++) {
    printf("%d\n",ans[j]);
  }
  return 0;
}
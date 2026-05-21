#include<stdio.h>
int main(void)
{
	int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	if(b/a>=c){
		printf("%d\n",c);
	}
	else if(c>b/a){
		printf("%d\n",b/a);
	}
	else if(a>b){
		printf("0\n");
	}
	return 0;
}
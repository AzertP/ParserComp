#include <stdio.h>
#include <string.h>
int main (){

int a,b;
char c[15],d[15],e[15];
	scanf("%s %s %d %d %s",&c,&d,&a,&b,&e);
if((strcmp(c,e)==0))
{
	printf("%d %d",a-1,b);
}
else{
	printf("%d %d",a,b-1);
}

	return 0;
}
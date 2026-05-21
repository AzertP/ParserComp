#include<stdio.h>
int main(void)
{
	int h1,h2,k1,k2,a,b,c,d,h,k,na,nb;
	scanf("%d %d %d %d %d %d %d %d",&h1,&h2,&k1,&k2,&a,&b,&c,&d);
	na=h1/10*c;
	nb=h2/10*b;
	h=h1*a+h2*b+h1/10*c+h2/20*d;
	na=k1/10*c;
	nb=k2/10*b;
	k=k1*a+k2*b+k1/10*c+k2/20*b;
	if(h>k){
	printf("hiroshi\n");
	}
	else if(h<k){
	printf("kenjiro\n");
	}
	else	{
	printf("even\n");
	}
	return 0;
}

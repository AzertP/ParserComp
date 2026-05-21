#include<stdio.h>
int main(){
	int w,h,n,i,c=0,d,e=0,f;
	int x[200],y[200],a[200];
	scanf("%d %d %d",&w,&h,&n);
	d=w;
	f=h;
	for(i=0;i<n;i++){
		scanf("%d %d %d",&x[i],&y[i],&a[i]);
		if(a[i]==1)
			if(c<x[i])
				c=x[i];
		if(a[i]==2)
			if(d>x[i])
				d=x[i];
		if(a[i]==3)
			if(e<y[i])
				e=y[i];
		if(a[i]==4)
			if(f>y[i])
				f=y[i];
	}
	if(c>=d || e>=f)
		printf("0\n");
	else
		printf("%d\n",(d-c)*(f-e));
	return 0;
}
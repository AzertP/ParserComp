#include <stdio.h>

void swap(int x[],int l,int r){
	int tmp;
	tmp=x[l];
	x[l]=x[r];
	x[r]=tmp;
}

void qsort(int x[],int l,int r){
	int i,j,p;
	i=l;
	j=r;
	p=x[(i+j)/2];
	while(1){
		while(p>x[i])i++;
		while(p<x[j])j--;
		if(i>=j)break;
		swap(x,i,j);
		i++;
		j--;
	}
	if(i-l>1)qsort(x,l,i-1);
	if(r-j>1)qsort(x,j+1,r);
}

int main(){
	int a[5];
	int i;
	for(i=0; i<5; i++){
		scanf("%d",&a[i]);
	}
	qsort(a,0,4);
	printf("%d %d %d %d %d\n",a[4],a[3],a[2],a[1],a[0]);

	return 0;
}
#include<stdio.h>


int main ()
{
	int N,a[100],flag;
	int i,j;
	int temp,count=0;

	scanf("%d",&N);

	for(i=0;i<N;i++){
		scanf("%d",&a[i]);
	}

	i = 0;
	for(i=0;i<N;i++){
		flag = 0;
		for(j=N-1;j>=i+1;j--){
			if(a[j] < a[j-1]){
				temp = a[j-1];
				a[j-1] = a[j];
				a[j] = temp;
				flag = 1;
				count++;
			}
		}
	}

	for(i=0;i<N;i++){
		printf("%d",a[i]);
		if(i == N-1)	printf("\n");
		else printf(" ");
	}
	printf("%d\n", count);

	return 0;
}
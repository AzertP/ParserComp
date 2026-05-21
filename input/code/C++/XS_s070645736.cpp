#include<stdio.h>
int main(){
	int A,B;
	1 <=A && A<=B && B<=20;
	scanf("%d%d",&A,&B);
	if(B%A==0){
		printf("\n%d",A+B);}else{
		printf("\n%d",B-A);
	}
	return 0;
}
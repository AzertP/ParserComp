#include <stdio.h>
#include <stdlib.h>
#include<time.h>

int main(void){
	int a,b,i,ans=0;
	scanf("%d",&a);
	scanf("%d",&b);
	for(i=0;i<=1;i++){
		if(a>=b){
			ans+=a;
			a--;

		}
		else{
			ans+=b;
			b--;

		}
	}
	printf("%d",ans);
	return 0;

}


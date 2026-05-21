#include<stdio.h>
#include<math.h>

int main(){
	int a[1145][2],n,s,i,j,cnt,num,total;
	for(i=0;i<1024;i++){
		for(j=1,cnt=0,num=0,total=0;cnt<10;j*=2,cnt++){
			if(i&j){
				total+=cnt;num++;
			}
		}
		a[i][0]=total;
		a[i][1]=num;
	}
	//for(i=0;i<1024;i++) printf("i=%d,total=%d,num=%d     ",i,a[i][0],a[i][1]);
	while(1){
		scanf("%d %d",&n,&s);
		if(n==0&&s==0) break;
		cnt=0;
		for(i=0;i<1024;i++){
			if(a[i][1]==n){
				if(a[i][0]==s)cnt++;
			}
		}
		printf("%d\n",cnt);
	}
	return 0;
}
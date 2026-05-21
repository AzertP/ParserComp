#include <stdio.h>
#include <string.h>

int N,M;
int C[16];
int x[20000];

/* [now position][prev value] */
int memo[20000][256];

int tansaku(int now_pos,int prev_value) {
	int result=0x7fffffff;
	int i;
	if(now_pos>=N)return 0;
	if(memo[now_pos][prev_value]>0)return memo[now_pos][prev_value]-1;
	for(i=0;i<M;i++) {
		int now_result;
		int now_sample=prev_value+C[i];
		if(now_sample<0)now_sample=0;
		if(now_sample>255)now_sample=255;
		now_result=(now_sample-x[now_pos])*(now_sample-x[now_pos]);
		now_result+=tansaku(now_pos+1,now_sample);
		if(now_result<result)result=now_result;
	}
	memo[now_pos][prev_value]=result+1;
	return result;
}

int main(void) {
	while(scanf("%d%d",&N,&M)==2 && (N|M)!=0) {
		int i;
		for(i=0;i<M;i++)scanf("%d",&C[i]);
		for(i=0;i<N;i++)scanf("%d",&x[i]);
		memset(memo,0,sizeof(memo));
		printf("%d\n",tansaku(0,128));
	}
	return 0;
}
#include <bits/stdc++.h>
#define file "gemo"
using namespace std;
typedef long long ll;
const int mod=1000000007;
int read(){
	char c=getchar();
	int x=0;
	while(c<'0' || c>'9')
		c=getchar();
	while(c>='0' && c<='9'){
		x=x*10+c-'0';
		c=getchar();
	}
	return x;
}
int a[15][15];
ll dp[1<<15][16],pw[2333];
int to[16][1<<15],qwq[1<<15];
int main(){
	//freopen(file".in","r",stdin);
	//freopen(file".out","w",stdout);
	int n=read(),m=read();
	pw[0]=1;
	for(int i=1;i<=m;i++)
		pw[i]=pw[i-1]*2%mod;
	for(int i=1;i<=m;i++){
		int u=read()-1,v=read()-1;
		a[u][v]=1;
	}
	for(int i=0;i<n;i++)
		for(int j=0;j<(1<<n);j++){
			if (j>>i&1)
				continue;
			for(int k=0;k<n;k++)
				if (j>>k&1)
					to[i][j]+=a[i][k];
			//fprintf(stderr,"to[%d][%d]=%d\n",i,j,to[i][j]);
		}
	for(int i=0;i<(1<<n);i++)
		for(int j=0;j<n;j++)
			if (!(i>>j&1))
				qwq[i]+=to[j][i];
	dp[0][0]=1;
	for(int i=0;i<(1<<n);i++){
		int s=(1<<n)-1-i;
		if ((i&1)!=(i>>1&1))
			continue;
		//fprintf(stderr,"i=%d\n",i);
		bool flag=true;
		for(int j=0;j<n;j++)
			if (dp[i][j])
				flag=false;
		if (flag)
			continue;
		for(int j=s;j;j=(j-1)&s){ //i->j ~to->j
			if ((j&1)!=(j>>1&1))
				continue;
			ll ans=1;
			int sm=0;
			int v=i|j;
			for(int o=0;o<n;o++)
				if (!(v>>o&1))
					ans=ans*((1<<to[o][j])-1)%mod,sm+=to[o][j];
			ans=ans*pw[qwq[j]-sm]%mod;
			if (!ans)
				continue;
			for(int o=0;o<n;o++){
				//fprintf(stderr,"dp[%d][%d]=%lld\n",i,o,dp[i][o]);
				dp[v][o+1]=(dp[v][o+1]+dp[i][o]*ans)%mod;
			}
		}
	}
	ll ans=0;
	for(int i=0;i<=n;i++)
		ans=(ans+dp[(1<<n)-1][i])%mod;
	printf("%lld",((pw[m]-ans)%mod+mod)%mod);
	//fprintf(stderr,"%.5f",(double)clock()/CLOCKS_PER_SEC);
}
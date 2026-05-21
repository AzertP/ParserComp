#include<bits/stdc++.h>
#define MOD 998244353
using namespace std;
long long cnt[100100];
int d[100100];
int n;
long long pow(long long a,long long b){
	long long res=1LL;
	while(b){
		if(b&1)
			res=(res*a)%MOD;
		a=(a*a)%MOD;
		b>>=1;
	}
	return res%MOD;
}
int main(){
	//freopen("input.txt","r",stdin);
	//freopen("output.txt","w",stdout);
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);
	cin>>n;
	for(int i=1;i<=n;i++)
		cin>>d[i];
	if(d[1]!=0){
		cout<<0<<endl;
		return 0;
	}
	for(int i=1;i<=n;i++)
		cnt[d[i]]+=1LL;
	if(cnt[0]>1){
		cout<<0<<endl;
		return 0;
	}
	long long res=1LL;
	for(int i=2;i<=n;i++){
		res*=pow(cnt[i-1],cnt[i]);
		res%=MOD;
	}
	cout<<res<<endl;
	return 0;
}

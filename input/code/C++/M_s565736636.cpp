#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define N 200000
#define mod1 1000000007
#define mod2 1000000009
#define mod3 998244353
#define endl '\n'
#define IO ios_base::sync_with_stdio(false);cin.tie(NULL);
void Google(ll tt){
    cout<<"Case #"<<tt<<": ";
}

int main()
{
    IO
	clock_t begin = clock();
    ll n,m;
    cin>>n>>m;
    ll dp[n+1][m+1];
    char ch[n+1][m+1];
    for(ll i=0;i<=n;i++)
    {
    	for(ll j=0;j<=m;j++)
    		dp[i][j] = 0;
    }
    for(ll i=1;i<=n;i++)
    {
    	for(ll j=1;j<=m;j++)
    	{
    		cin>>ch[i][j];
    	}
    }
    dp[1][1] = 1;
    for(ll i=1;i<=n;i++)
    {
    	for(ll j=1;j<=m;j++)
    	{
    		if(i==1 && j==1)
    			continue;
    		dp[i][j] = (dp[i-1][j] + dp[i][j-1])%mod1;
    		if(ch[i][j]=='#')
    			dp[i][j] = 0;
    	}
    }
    cout<<dp[n][m]<<endl;
    // cout<<double(clock() - begin)/CLOCKS_PER_SEC<<endl;
    return 0;
}
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template <typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

#define vi vector<int>
#define pi pair<int,int>
#define sz(a) a.size()
#define all(a) a.begin(),a.end()
#define F first
#define S second
#define pb push_back
#define eb emplace_back
#define ll long long 

template <typename A, typename B>
istream& operator>>(istream& input,pair<A,B>& x){
	input>>x.F>>x.S;
	return input;
}

template <typename A>
istream& operator>>(istream& input,vector<A>& x){
	for(auto& i:x)
		input>>i;
	return input;
}

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int mod=1e9+7;

int mul(int a,int b)
{
	return (a*1ll*b)%mod;
	}
	
	int add(int a,int b)
	{
		a+=b;
		if(a>=mod)return a-mod;
		return a;
		}
int powz(int a,int b)
{
	int res=1;
	while(b)
	{
		if(b&1)res=mul(res,a);
		b/=2;
		a=mul(a,a);
		}
		return res;
	}



void solve()
{
int n,k;
cin>>n>>k;
int dp[n+1],h[n]={0};
for(int i=0;i<=n;i++)dp[i]=1e9;
dp[0]=0;
cin>>h[0];
for(int i=1;i<n;i++)
{
	cin>>h[i];
	for(int j=1;j<=k;j++)
	{
		if(i<j)break;
	dp[i]=min(dp[i],dp[i-j]+abs(h[i]-h[i-j]));
	//	if(i>1)dp[i]=min(dp[i],dp[i-2]+abs(h[i]-h[i-2]));
}
	}	
	cout<<dp[n-1];
}
	
	
int main()
{   ios_base::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);
	int t=1;
	//cin>>t;
	while(t--)
	{
		solve();
		}
	
	
	}

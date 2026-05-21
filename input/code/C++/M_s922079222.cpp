#include<bits/stdc++.h>
using namespace std;

#define ll long long int
#define watch(x) cout << (#x) << " is " << (x) << endl
#define f(i,x,n)  for(int i=x;i<n;i++)
#define FASTIO cin.tie(0); cout.tie(0);
#define eb(x) emplace_back(x)
#define mp(a,b) make_pair(a,b)
#define sz(a) int((a).size())
#define mod 1000000007
#define tr(c,i) for(typeof((c)).begin() i = (c).begin(); i != (c).end(); i++)


int main()
{
	FASTIO;

	    ll n,m;
	     cin>>n>>m;
	     ll ans=0;

	     if(n>=1)
            ans=(n*(n-1))/2;
	     if(m>=1)
            ans+=(m*(m-1))/2;

	     cout<<ans<<"\n";


	return 0;
}




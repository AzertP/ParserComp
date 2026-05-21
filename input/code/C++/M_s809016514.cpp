#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double lf;
typedef pair<int,int> ii;
#define REP(i,n) for(int i=0;i<n;i++)
#define FILL(i,n) memset(i,n,sizeof(i));
#define X first
#define Y second
#define pb push_back
#ifdef brian
#define  debug(...) do{\
    fprintf(stderr,"%s - %d : (%s) = ",__PRETTY_FUNCTION__,__LINE__,#__VA_ARGS__);\
    _do(__VA_ARGS__);\
}while(0)
template<typename t> void _do(t &&x){cerr<<x<<endl;}
template<typename t,typename ...T> void _do(t &&x,T &&...tail){cerr<<x<<", ";_do(tail...);}
#else
#define debug(...)
#endif // brian


ll n,k;
int main()
{
    ios_base::sync_with_stdio(0);cin.tie(0);
    cin>>k>>n;
    ll ans=0;
    REP(i,k+1)
    {
        REP(j,k+1)
        {
            if(n-i-j>=0&&n-i-j<=k)ans++;
        }
    }
    cout<<ans<<endl;

}

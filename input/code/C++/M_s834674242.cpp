#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define rep(i, n) for (ll i = 0; i < n; i++)
#define Rep(i, r, n) for (ll i = r; i < n; i++)
#define debug(x) cout << #x << " = " << (x) << endl;
#define pb push_back
#define MOD 1000000007
//#define MOD 998244353
//define MOD 1000000007000000
#define INF 1000000000000
#define EPS 0.00000000001
typedef long long ll;

int main()
{
    cin.tie(0);
    ios::sync_with_stdio(false);
    cout << fixed << setprecision(10);
    //---------------------------------------------

    int n;
    string s;
    cin >> n >> s;
    if(n==1){
        cout<<1<<endl;
    }
    else{

    

    sort(all(s));
    ll memo=1;
    ll ans=1;
    rep(i,n){
        if(i==0){
            continue;
        }
        if(s[i]==s[i-1]){
            memo++;
        }
        else{
            memo++;
            ans*=memo;
            ans%=MOD;
            memo=1;
        }

        if(i==n-1){
            memo++;
            ans*=memo;
            ans%=MOD;
            memo=1;
        }

    }
    cout<<ans-1<<endl;
    }
    
}

#include<bits/stdc++.h>
#define REP(i,n) for(int i=0;i<(n);i++)
#define ALL(v) (v).begin(),(v).end()
#define int long long
using namespace std;
typedef vector<int>   vint;
typedef pair<int,int> pint;

signed main()
{   
    vint a(3);
    REP(i,3) cin>>a[i];
    sort(ALL(a));
    int K; cin>>K;
    cout<<a[0]+a[1]+(a[2]<<K)<<endl;
}

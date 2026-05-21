#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<algorithm>
#include <numeric>
#include<map>
#include<unordered_map>
#include <queue>
 
using namespace std;
using ll=long long;
#define rep(i,n) for(ll i=0;i<n;++i)
#define all_map(itr,mp) for(auto itr=mp.begin();itr!=mp.end();itr++)
#define ALL(a) (a).begin(),(a).end()

int main(){
    ll n;
    cin >> n;
    ll dict[100010] = {};
    ll mx = 0;
    rep(i, n){
        ll a;
        cin >> a;
        dict[a]++;
        mx = max(a, mx);
    }
    
    ll sum, cnt, knd;
    sum = cnt = knd = 0;
    rep(i, mx+1){
        if(dict[i] > 1)cnt++, sum+=dict[i];
        if(dict[i] > 0)knd++;
    }
    if((sum-cnt)%2 != 0)cout << knd-1 << endl;
    else cout << knd << endl;
    
    
}
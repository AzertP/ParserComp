#pragma GCC optimize("Ofast")
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <bitset>
#include <cmath>
#include <limits>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
using namespace std;
#define INF 1LL << 30
#define MAX 100000
#define MOD 1000000007
typedef long long ll;
typedef pair<int,int> P;
//typedef pair<pair<int,int>,int> p;
#define bit(n, k) ((n >> k) & 1) /*nのk bit目*/
#define rad_to_deg(rad) (((rad) / 2 / M_PI) * 360)
struct edge
{
    ll to, cost, val;
};
template <class T, class U>
bool chmin(T &a, const U &b)
{
    if (a <= b)
        return false;
    a = b;
    return true;
}
template <class T, class U>
bool chmax(T &a, const U &b)
{
    if (a >= b)
        return false;
    a = b;
    return true;
}
//__builtin_popcount(S);
//C.erase(unique(C.begin(),C.end()),C.end());
//#define int ll

int d[210000];

signed main(void)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin>>N;
    vector<int> V[210000];
    for(int i=0;i<N-1;i++){
        int a,b;
        cin>>a>>b;
        V[a].push_back(b);
        V[b].push_back(a);
    }
    d[1]=0;
    queue<int> que;
    que.push(1);
    int t;
    while(!que.empty()){
        int now=que.front(); que.pop();
        for(int next:V[now]){
            if(d[next]==0){
                d[next]=d[now]+1;
                t=next;
                que.push(next);
            }
        }
    }
    for(int i=0;i<210000;i++) d[i]=0;
    que.push(t);
    int ans;
    while(!que.empty()){
        int now=que.front(); que.pop();
        for(int next:V[now]){
            if(d[next]==0){
                d[next]=d[now]+1;
                t=next;
                ans=d[next];
                que.push(next);
            }
        }
    }
    ans++;
    //cout<<ans<<endl;
    if(ans%3==2) cout<<"Second"<<endl;
    else cout<<"First"<<endl;
}

#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>
#include <bitset>
#include <cmath>
#include <queue>
#include <tuple>
#include <deque>
#include <stack>
#include <list>
#include <map>
#include <set>
using namespace std;

using ll=long long;

typedef long long ll;
typedef pair<int,int> ii;
typedef pair<string,int> si;
typedef pair<int,ii> iii;
typedef vector <si> vsi;
typedef vector <ii> vii;
typedef vector <int> vi;
typedef vector <ll> vll;
typedef vector <vll> vl2d;
typedef vector <vi> vi2d;
typedef vector <char> vc;
typedef vector <bool> vb;
typedef vector <string> vs;
typedef map <string,vs> msvs;
typedef map <string,int> msi;
typedef map <char,int> mci;
typedef map <int,int> mii;
typedef map <long,long> ml;
typedef map <string,string> mss;
#define rep(i,a,n) for (ll i=a;i<n;i++)
#define per(i,a,n) for (ll i=n-1;i>=a;i--)
#define foreach(x,arr) for(auto& x:arr)
#define all(x) (x).begin(), (x).end()
#define pb push_back
#define mp make_pair
#define endl '\n' // Normal `cout << endl` flushes the output every time wich hit performance badly
#define INF 1000000000
#define MOD 1000000007
#define deb(x) cout<<#x<<": "<<x<<endl;
#define printArray(arr) for(auto x: arr) { cout<<x<<"\n"; }
#define printMatrix(mat) for(auto x: mat) { cout<<" "; printArray(x); cout<<endl; }
#define printMap(mmap) for(auto p: mmap) { cout<< p.first<<": "<<p.second<<endl; }
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define forch(ch,str) for(char& c : str)
#define FILL(a,b) memset(a,b,sizeof(a))
//Swift related stuff
#define append push_back
#define nil nullptr
#define guard(_condition) if (bool(_condition)){}
#define count size()
#define let auto
#define var auto
#define startsWith(str) rfind(str, 0)
#define prq priority_queue
#define tostr(argument) to_string(argument)
#define alphabet "abcdefghijklmnopqrstuvwxyz"

void join(const vector<string>& v, char c, string& s) {

   s.clear();

   for (vector<string>::const_iterator p = v.begin();
        p != v.end(); ++p) {
      s += *p;
      if (p != v.end() - 1)
        s += c;
   }
}

vll arr;
vector<vl2d> dp(3001, vl2d(3001, vll(2, -1)));
ll n;
ll jiro(ll start, ll end);

ll taro(ll start, ll end) {
    guard(start <= end) else {
        return 0;
    }
    guard(dp[start][end][0] == -1) else {
        return dp[start][end][0];
    }
    
    dp[start][end][0] = MAX(
                            arr[start] + (start + 1 < arr.count ? jiro(start+1, end) : 0),
                            arr[end] + (end-1 >= 0 ? jiro(start, end-1) : 0)
                            );
    return dp[start][end][0];
    
    
}

ll jiro(ll start, ll end) {
    guard(start <= end) else {
        return 0;
    }
    guard(dp[start][end][1] == -1) else {
        return dp[start][end][1];
    }
    
    dp[start][end][1] = MIN(
                            -arr[start] + (start + 1 < arr.count ? taro(start+1, end) : 0),
                            -arr[end] + (end-1 >= 0 ? taro(start, end-1) : 0)
                            );
    
    return dp[start][end][1];
}

int main(){
    ios_base::sync_with_stdio(false); cin.tie(NULL); // Fast input
    
    cin>>n;
    rep(i, 0, n) {
        int num;
        cin>>num;
        arr.append(num);
    }
    cout<<taro(0,n-1)<<endl;
    return 0;
}

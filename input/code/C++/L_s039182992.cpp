
// Problem: 
// 			E - Multiplication 4
// 			Editorial
// 		
// Contest: AtCoder - AtCoder Beginner Contest 173
// URL: https://atcoder.jp/contests/abc173/tasks/abc173_e
// Memory Limit: 1024 MB
// Time Limit: 2000 ms
// Powered by CP Editor (https://github.com/cpeditor/cpeditor)

//#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")
//
//#include <immintrin.h>
//#include <emmintrin.h>

#include <bits/stdc++.h>
//#pragma GCC optimize("O2")
#define vi vector<int>
#define pii pair<int, int >
#define mp make_pair
#define fi first
#define se second
#define pb push_back
#define LL long long
#define rep(i,a,n) for (int i=a;i<=n;i++)
#define per(i,a,n) for (int i=n;i>=a;i--)
#define all(x) (x).begin(), (x).end()
#define all2(x,n) (x+1), (x+1+n)
#define sz(x) ((int)(x).size())
#define mod(x) ((x)%MOD)
#define debug(x) cerr<<#x<<" : "<<x<<endl
#define mt make_tuple
#define eb emplace_back
#define o(X) (1<<(X))
#define oL(X) (1LL<<(X))
#define contain(S,X) (((S)&o(X))!=0)
#define containL(S,X) (((S)&oL(X))!=0)
#define ppt(x) __builtin_popcount(x)
using namespace std;
const int INF=0x3f3f3f3f,N=1e6+5,MOD=1e9+7;
const LL INF_LL=0x3f3f3f3f3f3f3f3fLL;
inline int getplc(int x,int y) { return (x>>y)&1; }
template<typename T>
T square(T x) {return x*x;}
LL qpow(LL a,LL b=MOD-2,LL _MOD=MOD){
	LL res=1;
	for(;b;b>>=1,a=a*a%_MOD){
		if(b&1)res=res*a%_MOD;
	}
	return res;
}
// Smax
//int Smax() { return -INF; }
template <typename T>
T Smax(T x) { return x; }
template<typename T, typename... Args>
T Smax(T a, Args... args) { return max(a, Smax(args...)); }
// Smin
template <typename T>
T Smin(T x) { return x; }
template<typename T, typename... Args>
T Smin(T a, Args... args) { return min(a, Smin(args...)); }
template <typename T>
// erro
#define errorl(args...) { string _s = #args; replace(_s.begin(), _s.end(), ',', ' '); stringstream _ss(_s); istream_iterator<string> _it(_ss); errl(_it, args); }

void errl(istream_iterator<string> it) {}
template<typename T, typename... Args>
void errl(istream_iterator<string> it, T a, Args... args) {
	cerr << *it << " = " << a << endl;
	errl(++it, args...);
}

#define error(args...) { string _s = #args; replace(_s.begin(), _s.end(), ',', ' '); stringstream _ss(_s); istream_iterator<string> _it(_ss); err(_it, args); cerr<<endl;}
void err(istream_iterator<string> it) {}
template<typename T, typename... Args>
void err(istream_iterator<string> it, T a, Args... args) {
	cerr << *it << "=" << a << " # ";
	err(++it, args...);
}
void Solve();
int main() {
#ifndef ONLINE_JUDGE
//	freopen("in.txt","r",stdin);
//    freopen("o1.txt","w",stdout);
#endif
	ios::sync_with_stdio(false);cin.tie(0),cout.tie(0);
	Solve();
	return 0;
}

//////////////////////////////////////////////////////////////////

int a[N],k;
int cal(){
  int ans=1;
  rep(i,1,k)ans=((LL)ans*a[i])%MOD;
  ans=((LL)ans+MOD)%MOD;
  return ans;
}

void Solve(){
  int n,zero=0,ans=0;
  cin>>n>>k;
  rep(i,1,n){
    cin>>a[i];
    zero|=(a[i]==0);
  }
  sort(a+1,a+1+n,[&](int a,int b){
    return abs(a)>abs(b);
  });
  int flg=1;
  rep(i,1,k)if(a[i]<0)flg=-flg;
  if(flg>0){
    cout<<cal()<<endl;
    return;
  }
  int pos=-1,neg=-1,pos2=-1,neg2=-1;
  per(i,1,k){
    if(pos==-1&&a[i]>0)pos=i;
    if(neg==-1&&a[i]<0)neg=i;
  }
  rep(i,k+1,n){
    if(pos2==-1&&a[i]>0)pos2=i;
    if(neg2==-1&&a[i]<0)neg2=i;
  }
  if(~pos&&~neg&&~pos2&&~neg2){
    if((LL)a[pos2]*a[pos]<(LL)a[neg]*a[neg2]){
      swap(a[pos],a[neg2]);
    }
    else swap(a[neg],a[pos2]);
  }
  else if(~pos&&~neg2)swap(a[pos],a[neg2]);
  else if(~pos2&&~neg)swap(a[neg],a[pos2]);
  else{
    if(zero){
      ans=0;
      cout<<ans<<endl;
      return;
    }
    else reverse(a+1,a+1+n);
  }
  cout<<cal()<<endl;
  return;
} 

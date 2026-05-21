/*ver 5*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <queue>
#include <tuple>
#include <map>
#include <list>
#include <set>
using namespace std;
void init() {cin.tie(0);ios::sync_with_stdio(false);cout << fixed << setprecision(15);}
using ll = long long;
using ld = long double;
using vl = vector<ll>;
using vd = vector<ld>;
using vs = vector<string>;
using vb = vector<bool>;
using vvl = vector<vector<ll>>;
using vvd = vector<vector<ld>>;
using vvs = vector<vector<string>>;
using vvb = vector<vector<bool>>;
using pll = pair<ll,ll>;
using mll = map<ll,ll>;
#define EACH(x,v) for(auto& x : v)
#define FOR(i,a,b) for(ll i=(ll)a;i<(ll)b;i++)
#define REFOR(i,a,b) for(ll i=(ll)b-1;i>=(ll)a;i--)
#define REP(i,n) for(ll i=0;i<(ll)n;i++)
#define REREP(i,n) for(ll i=(ll)n-1;i>=0;i--)
#define pb push_back
#define eb emplace_back
#define fi first
#define se second
#define mp make_pair
const ll INF = 1LL << 60;
#define CLR(mat,f) memset(mat, f, sizeof(mat))
#define IN(a, b, x) (a<=x&&x<=b)
#define UNIQUE(v) v.erase( unique(v.begin(), v.end()), v.end() ) //被り削除
#define debug cout << "line : " << __LINE__ << " debug" << endl;
#define ini(...) int __VA_ARGS__; in(__VA_ARGS__)
#define inl(...) long long __VA_ARGS__; in(__VA_ARGS__)
#define ind(...) long double __VA_ARGS__; in(__VA_ARGS__)
#define ins(...) string __VA_ARGS__; in(__VA_ARGS__)
#define inc(...) char __VA_ARGS__; in(__VA_ARGS__)
void in(){}
template <typename T,class... U> void in(T &t,U &...u){ cin >> t; in(u...);}
void out(){cout << endl;}
template <typename T,class... U> void out(const T &t,const U &...u){ cout << t; if(sizeof...(u)) cout << " "; out(u...);}
template <typename T> void die(T x){out(x); exit(0);}
template <typename T> void in1(T &s) {REP(i,s.size()){in(s[i]);}}
template <typename T> void in2(T &s,T &t) {REP(i,s.size()){in(s[i],t[i]);}}
template <typename T> void in3(T &s,T &t,T &u) {REP(i,s.size()){in(s[i],t[i],u[i]);}}
template <typename T> void in4(T &s,T &t,T &u,T &v) {REP(i,s.size()){in(s[i],t[i],u[i],v[i]);}}
template <typename T> void in5(T &s,T &t,T &u,T &v,T &w) {REP(i,s.size()){in(s[i],t[i],u[i],v[i],w[i]);}}
template <typename T> void outv(T s) {REP(i,s.size()){cout<<s[i]<<" ";}cout<<endl;}
template <typename T> void out1(T s) {REP(i,s.size()){out(s[i]);}}
template <typename T> void out2(T s,T t) {REP(i,s.size()){out(s[i],t[i]);}}
template <typename T> void out3(T s,T t,T u) {REP(i,s.size()){out(s[i],t[i],u[i]);}}
template <typename T> void out4(T s,T t,T u,T v) {REP(i,s.size()){out(s[i],t[i],u[i],v[i]);}}
template <typename T> void out5(T s,T t,T u,T v,T w) {REP(i,s.size()){out(s[i],t[i],u[i],v[i],w[i]);}}
#define all(v) (v).begin(),(v).end()
#define rall(v) (v).rbegin(),(v).rend()
template <typename T> T allSum(vector<T> a){T ans=T();EACH(it,a)ans+=it;return ans;}
ll ceilDiv(ll a,ll b) {return (a+b-1)/b;}
ld dist(pair<ld,ld> a, pair<ld,ld> b){return sqrt(abs(a.fi-b.fi)*abs(a.fi-b.fi)+abs(a.se-b.se)*abs(a.se-b.se));} // ２点間の距離 
ll gcd(ll a, ll b) { return b != 0 ? gcd(b, a % b) : a; }
ll lcm(ll a,ll b){ return a / gcd(a,b) * b;}
template <class A, class B> inline bool chmax(A &a, const B &b) { return b > a && (a = b, true); }
template <class A, class B> inline bool chmin(A &a, const B &b) { return b < a && (a = b, true); }
#define YES(n) ((n) ? "YES" : "NO"  )
#define Yes(n) ((n) ? "Yes" : "No"  )
#define yes(n) ((n) ? "yes" : "no"  )
const int mod = 1000000007;
class mint {
	long long x;
public:
	mint(long long x=0) : x((x%mod+mod)%mod) {}
	mint operator-() const { 
		return mint(-x);
	}
	mint& operator+=(const mint& a) {
		if ((x += a.x) >= mod) x -= mod;
		return *this;
	}
	mint& operator-=(const mint& a) {
		if ((x += mod-a.x) >= mod) x -= mod;
		return *this;
	}
	mint& operator*=(const  mint& a) {
		(x *= a.x) %= mod;
		return *this;
	}
	mint operator+(const mint& a) const {
		mint res(*this);
		return res+=a;
	}
	mint operator-(const mint& a) const {
		mint res(*this);
		return res-=a;
	}
	mint operator*(const mint& a) const {
		mint res(*this);
		return res*=a;
	}
	mint pow(ll t) const {
		if (!t) return 1;
		mint a = pow(t>>1);
		a *= a;
		if (t&1) a *= *this;
		return a;
	}
	// for prime mod
	mint inv() const {
		return pow(mod-2);
	}
	mint& operator/=(const mint& a) {
		return (*this) *= a.inv();
	}
	mint operator/(const mint& a) const {
		mint res(*this);
		return res/=a;
	}
	friend ostream& operator<<(ostream& os, const mint& m){
		os << m.x;
		return os;
	}
};
class UnionFind {
public:
	vector<ll> par; // par[i]:iの親の番号　(例) par[3] = 2 : 3の親が2
	vector<ll> siz; // siz[i]:iのサイズ　　(例) siz[3] = 2 : 3のサイズが2
	UnionFind(ll N) : par(N),siz(N,1LL) { //最初は全てが根であるとして初期化
	for(int i = 0; i < N; i++) par[i] = i;
	}
	ll root(ll x) { // データxが属する木の根を再帰で得る：root(x) = {xの木の根}
		if (par[x] == x) return x;
		return par[x] = root(par[x]);
	}
	bool unite(ll x, ll y) { // xとyの木を併合
		x = root(x); //xの根をrx
		y = root(y); //yの根をry
		if (x == y) return false; //xとyの根が同じ(=同じ木にある)時はそのまま
		if (siz[x] < siz[y]) swap(x , y);
		siz[x] += siz[y];
		par[y] = x; //xとyの根が同じでない(=同じ木にない)時：xの根rxをyの根ryにつける
		return true;
	}
	bool same(int x, int y) { // 2つのデータx, yが属する木が同じならtrueを返す
		return root(x) == root(y);
	}
	ll size(ll x) { // 素集合のサイズ
		return siz[root(x)];
	}
};
class Dag {
public:
	vvl g;
	vl inCnt;
	vl sorted;
	bool sortedF=false;
    Dag(ll N) : g(N),inCnt(N){
    }
	void input(ll x,ll y){
		g[x].pb(y);
		inCnt[y]++;
	}
	bool tpSort(){
		vl tmp=inCnt;
		deque<ll> dq;
		vl ans;
		REP(i,g.size())if(tmp[i]==0)dq.pb(i);
		while(dq.size()!=0){
			ll v=dq.front();
			dq.pop_front();
			ans.pb(v);
			EACH(it,g[v]){
				tmp[it]--;
				if(tmp[it]==0)dq.pb(it);
			}
		}
		sorted.swap(ans);
		sortedF=true;
		return sorted.size()==g.size();
	}
	ll longest(){
		if(sortedF==false)tpSort();
		vl dp(g.size(),0);
		ll res=0;
		EACH(it1,sorted){
			EACH(it2,g[it1]){
				chmax(dp[it2],dp[it1]+1);
			}
			chmax(res,dp[it1]);
		}
		return res;
	}
};

int main(){
	init();
	inl(n);
	vector<unsigned long long> a(n);
	in1(a);
	unsigned long long sum=0;
	REP(i,n){
		sum^=a[i];
	}
	vector<unsigned long long> ans(n);
	REP(i,n){
		ans[i]=sum^a[i];
	}
	outv(ans);
	return 0;
}
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <cmath>
#include <map>
#include <queue>
#include <deque>
#include <stack>
#include <set>
#include <stdio.h>
#include <tuple>
#include <bitset>
#include <cfloat>
#include <fstream>
#include <limits.h>
#include <list>
#include <math.h>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <utility>
#include <sstream>
#include <cassert>
#include <chrono>
template<class T> inline bool chmin(T& a, T b) {
	if (a > b) {
		a = b;
		return true;
	}
	return false;
}
template<class T> inline bool chmax(T& a, T b) {
	if (a < b) {
		a = b;
		return true;
	}
	return false;
}
using namespace std;
#define int long long
#define ll long long
#define rep(i, n) for (ll i = 0; i < (n); i++)
#define P pair<ll, ll>
#define sz(x) (ll)x.size()
#define ALL(x) (x).begin(),(x).end()
#define ALLR(x) (x).rbegin(),(x).rend()
#define VE vector<ll>
#define COUT(x) cout<<(x)<<endl
#define MA map<ll,ll>
#define SE set<ll>
#define PQ priority_queue<ll>
#define PQR priority_queue<ll,VE,greater<ll>>
#define YES(n) cout << ((n) ? "YES" : "NO"  ) << endl
#define Yes(n) cout << ((n) ? "Yes" : "No"  ) << endl
#define EPS (1e-10)
#define pb push_back
long long MOD = 1000000007;
//long long MOD = 998244353;
const long long INF = 1LL << 60;
const long double PI = acos(-1.0);
struct mint {
	ll x; // typedef long long ll;
	mint(ll x = 0) :x((x%MOD + MOD) % MOD) {}
	mint operator-() const { return mint(-x); }
	mint& operator+=(const mint a) {
		if ((x += a.x) >= MOD) x -= MOD;
		return *this;
	}
	mint& operator-=(const mint a) {
		if ((x += MOD - a.x) >= MOD) x -= MOD;
		return *this;
	}
	mint& operator*=(const mint a) {
		(x *= a.x) %= MOD;
		return *this;
	}
	mint operator+(const mint a) const {
		mint res(*this);
		return res += a;
	}
	mint operator-(const mint a) const {
		mint res(*this);
		return res -= a;
	}
	mint operator*(const mint a) const {
		mint res(*this);
		return res *= a;
	}
	mint pow(ll t) const {
		if (!t) return 1;
		mint a = pow(t >> 1);
		a *= a;
		if (t & 1) a *= *this;
		return a;
	}
	// for prime MOD
	mint inv() const {
		return pow(MOD - 2);
	}
	mint& operator/=(const mint a) {
		return (*this) *= a.inv();
	}
	mint operator/(const mint a) const {
		mint res(*this);
		return res /= a;
	}
	bool operator==(const mint a) const {
		return x == a.x;
	}
	bool operator!=(const mint a) const {
		return x != a.x;
	}
};
istream& operator>>(istream& is, mint& a) { return is >> a.x; }
ostream& operator<<(ostream& os, const mint& a) { return os << a.x; }
struct Sieve {
	int n;
	vector<int> f, primes;

	Sieve(int n = 1) :n(n), f(n + 1) {
		f[0] = f[1] = -1;
		for (ll i = 2; i <= n; ++i) {
			if (f[i]) continue;
			primes.push_back(i);
			f[i] = i;
			for (ll j = i * i; j <= n; j += i) {
				if (!f[j]) f[j] = i;
			}
		}
	}

	bool isPrime(int x) { return f[x] == x; }

	vector<int> factorList(int x) {
		vector<int> res;
		while (x != 1) {
			res.push_back(f[x]);
			x /= f[x];
		}
		return res;
	}

	vector<P> factor(int x) {
		vector<int> fl = factorList(x);
		if (fl.size() == 0) return {};
		vector<P> res(1, P(fl[0], 0));
		for (int p : fl) {
			if (res.back().first == p) {
				res.back().second++;
			}
			else {
				res.emplace_back(p, 1);
			}
		}
		return res;
	}
};
template<class t> t gcd(t a, t b) {
	return b != 0 ? gcd(b, a % b) : a;
}
ll lcm(ll a, ll b) {
	ll g = gcd(a, b);
	return a / g * b;
}
bool prime(ll n) {
	for (ll i = 2; i <= sqrt(n); i++) {
		if (n%i == 0)return false;
	}
	return n != 1;
}
map<ll, ll> prime_factor(ll n) {
	map<ll, ll> ret;
	for (ll i = 2; i * i <= n; i++) {
		while (n % i == 0) {
			ret[i]++;
			n /= i;
		}
	}
	if (n != 1) ret[n] = 1;
	return ret;
}
ll modinv(ll a, ll m) {
	ll b = m, u = 1, v = 0;
	while (b) {
		ll t = a / b;
		a -= t * b; swap(a, b);
		u -= t * v; swap(u, v);
	}
	u %= m;
	if (u < 0) u += m;
	return u;
}
vector<pair<char, int>> RunLength(string s) {
	if (s.size() == 0)return {};
	vector<pair<char, int>>res(1, pair<char, int>(s[0], 0));
	for (char p : s) {
		if (res.back().first == p) {
			res.back().second++;
		}
		else {
			res.emplace_back(p, 1);
		}
	}
	return res;
}
// Digit Count
int GetDigit(int num) {
	return log10(num) + 1;
}
// bit calculation[how many "1"] (= __builtin_popcount())
int bit_count(int n) {
	int cnt = 0;
	while (n > 0) {
		if (n % 2 == 1)cnt++;
		n /= 2;
	}
	return cnt;
}
vector<long long> enum_divisors(long long N) {
	vector<long long> res;
	for (long long i = 1; i * i <= N; ++i) {
		if (N % i == 0) {
			res.push_back(i);
			if (N / i != i) res.push_back(N / i);
		}
	}
	sort(res.begin(), res.end());
	return res;
}
const ll dx[4] = { 1,0,-1,0 };
const ll dy[4] = { 0,1,0,-1 };
//struct edge { ll to, cost; };
typedef long double ld;
struct UnionFind {
	vector <ll> par; // 各元の親を表す配列
	vector <ll> siz; // 素集合のサイズを表す配列(1 で初期化)
	UnionFind() {}
	// Constructor
	UnionFind(ll sz_) : par(sz_), siz(sz_, 1) {
		for (ll i = 0; i < sz_; ++i) par[i] = i; // 初期では親は自分自身
	}
	void init(ll sz_) {
		par.resize(sz_);
		siz.resize(sz_, 1);
		for (ll i = 0; i < sz_; ++i) par[i] = i; // 初期では親は自分自身
	}

	// Member Function
	// Find
	ll root(ll x) { // 根の検索
		while (par[x] != x) {
			x = par[x] = par[par[x]]; // x の親の親を x の親とする
		}
		return x;
	}

	// Union(Unite, Merge)
	bool merge(ll x, ll y) {
		x = root(x);
		y = root(y);
		if (x == y) return false;
		// merge technique（データ構造をマージするテク．小を大にくっつける）
		if (siz[x] < siz[y]) swap(x, y);
		siz[x] += siz[y];
		par[y] = x;
		return true;
	}

	bool issame(ll x, ll y) { // 連結判定
		return root(x) == root(y);
	}

	ll size(ll x) { // 素集合のサイズ
		return siz[root(x)];
	}
};
struct combination {
	vector<mint> fact, ifact;
	combination(int n) :fact(n + 1), ifact(n + 1) {
		//assert(n < mod);
		fact[0] = 1;
		for (int i = 1; i <= n; ++i) fact[i] = fact[i - 1] * i;
		ifact[n] = fact[n].inv();
		for (int i = n; i >= 1; --i) ifact[i - 1] = ifact[i] * i;
	}
	mint operator()(int n, int k) {
		if (k < 0 || k > n) return 0;
		return fact[n] * ifact[k] * ifact[n - k];
	}
	mint nPk(int n, int k) {
		return fact[n] * ifact[n - k];
	}
};
struct edge { int to, cost; };
using Graph = vector<vector<int>>;

/*
・MODあってる？？？
*/
int c;
VE sub(string s) {
	int n = s.size();
	int cnt = 0, last = -c - 1;
	VE res(n + 1);
	rep(i, n) {
		if (i - last > c && s[i] == 'o') {
			cnt++;
			last = i;
		}
		res[i + 1] = cnt;
	}
	return res;
}
signed main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	//cout << fixed << setprecision(20);
	//combination com(500010);
	int n, k; cin >> n >> k >> c;
	string s; cin >> s;
	auto left = sub(s);
	string t = s; reverse(ALL(t));
	auto right = sub(t);
	rep(i, n) {
		if (s[i] == 'o') {
			if (left[i] + right[n - i - 1] < k)cout << i + 1 << endl;
		}
	}
	return 0;
}

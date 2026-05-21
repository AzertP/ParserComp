#include<iostream>
#include<string>
#include<cstdio>
#include<vector>
#include<cmath>
#include<algorithm>
#include<functional>
#include<iomanip>
#include<queue>
#include<ciso646>
#include<random>
#include<map>
#include<set>
#include<bitset>
#include<stack>
#include<unordered_map>
#include<utility>
#include<cassert>
#include<complex>
#include<numeric>
using namespace std;

//#define int long long
typedef long long ll;

typedef unsigned long long ul;
typedef unsigned int ui;
constexpr ll mod = 1000000007;
const ll INF = mod * mod;
typedef pair<int, int>P;
#define stop char nyaa;cin>>nyaa;
#define rep(i,n) for(int i=0;i<n;i++)
#define per(i,n) for(int i=n-1;i>=0;i--)
#define Rep(i,sta,n) for(int i=sta;i<n;i++)
#define rep1(i,n) for(int i=1;i<=n;i++)
#define per1(i,n) for(int i=n;i>=1;i--)
#define Rep1(i,sta,n) for(int i=sta;i<=n;i++)
#define all(v) (v).begin(),(v).end()
typedef pair<ll, ll> LP;
typedef long double ld;
typedef pair<ld, ld> LDP;
const ld eps = 1e-12;
const ld pi = acos(-1.0);

void chmin(ll &a, ll b) {
	a = min(a, b);
}
void chmax(int &a, int b) {
	a = max(a, b);
}
ll mod_pow(ll a, ll n, ll m = mod) {
	a %= m;
	ll res = 1;
	while (n) {
		if (n & 1)res = res * a%m;
		a = a * a%m; n >>= 1;
	}
	return res;
}

struct modint {
	ll n;
	modint() :n(0) { ; }
	modint(ll m) :n(m) {
		if (n >= mod)n %= mod;
		else if (n < 0)n = (n%mod + mod) % mod;
	}
	operator int() { return n; }
};
bool operator==(modint a, modint b) { return a.n == b.n; }
modint operator+=(modint &a, modint b) { a.n += b.n; if (a.n >= mod)a.n -= mod; return a; }
modint operator-=(modint &a, modint b) { a.n -= b.n; if (a.n < 0)a.n += mod; return a; }
modint operator*=(modint &a, modint b) { a.n = ((ll)a.n*b.n) % mod; return a; }
modint operator+(modint a, modint b) { return a += b; }
modint operator-(modint a, modint b) { return a -= b; }
modint operator*(modint a, modint b) { return a *= b; }
modint operator^(modint a, int n) {
	if (n == 0)return modint(1);
	modint res = (a*a) ^ (n / 2);
	if (n % 2)res = res * a;
	return res;
}

ll inv(ll a, ll p) {
	return (a == 1 ? 1 : (1 - p * inv(p%a, a)) / a + p);
}
modint operator/(modint a, modint b) { return a * modint(inv(b, mod)); }
const int max_n = 1 << 18;
modint fact[max_n], factinv[max_n];
void init_f() {
	fact[0] = modint(1);
	for (int i = 0; i < max_n - 1; i++) {
		fact[i + 1] = fact[i] * modint(i + 1);
	}
	factinv[max_n - 1] = modint(1) / fact[max_n - 1];
	for (int i = max_n - 2; i >= 0; i--) {
		factinv[i] = factinv[i + 1] * modint(i + 1);
	}
}
modint comb(int a, int b) {
	if (a < 0 || b < 0 || a < b)return 0;
	return fact[a] * factinv[b] * factinv[a - b];
}

int dx[4] = { 0,1,0,-1 };
int dy[4] = { 1,0,-1,0 };

struct SegT {
private:
	int n; vector<int> node;vector<int> lazy;
	const int init_c = mod;
public:
	void init(int sz) {
		n = 1;
		while (n < sz)n <<= 1;
		node.resize(2 * n - 1, 0);
		lazy.resize(2 * n - 1, 0);
	}
	int f(int a, int b) {
		return min(a, b);
	}
	void eval(int k, int l, int r) {
		node[k] += lazy[k];
		if (r - l > 1) {
			lazy[2 * k + 1] += lazy[k];
			lazy[2 * k + 2] += lazy[k];
		}
		lazy[k] = 0;
	}
	void add(ll x, int a, int b, int k = 0, int l = 0, int r = -1) {
		if (r < 0)r = n;
		eval(k, l, r);
		if (r <= a || b <= l)return;
		if (a <= l && r <= b) {
			lazy[k] += x; eval(k, l, r);
		}
		else {
			add(x, a, b, k * 2 + 1, l, (l + r) / 2);
			add(x, a, b, k * 2 + 2, (l + r) / 2, r);
			node[k] = f(node[k * 2 + 1], node[k * 2 + 2]);
		}
	}
	int query(int a, int b, int k = 0, int l = 0, int r = -1) {
		if (r < 0)r = n;
		eval(k, l, r);
		if (r <= a || b <= l)return init_c;
		if (a <= l && r <= b)return node[k];
		else {
			int vl = query(a, b, k * 2 + 1, l, (l + r) / 2);
			int vr = query(a, b, k * 2 + 2, (l + r) / 2, r);
			return f(vl, vr);
		}
	}
	void onefind(vector<int> &res,int a, int b,int k=0,int l=0,int r=-1) {
		if (r < 0)r = n;
		eval(k, l, r);
		if (r <= a || b <= l)return;
		if (a <= l && r <= b) {
			if (node[k] > 1)return;
			if (l + 1 == r) {
				if(node[k]==1)res.push_back(k - (n - 1));
				return;
			}
		}
		onefind(res, a, b, 2 * k + 1, l, (l + r) / 2);
		onefind(res, a, b, 2 * k + 2, (l + r) / 2, r);
	}
};

struct SegT2 {
private:
	int n; vector<ll> node, lazy;
	const ll init_c = 0;
public:
	void init(int sz) {
		n = 1;
		while (n < sz)n <<= 1;
		node.resize(2 * n - 1, init_c);
		lazy.resize(2 * n - 1, 0);
	}
	ll f(ll a, ll b) {
		return a+b;
	}
	void eval(int k, int l, int r) {
		node[k] += lazy[k];
		if (r - l > 1) {
			lazy[2 * k + 1] += lazy[k];
			lazy[2 * k + 2] += lazy[k];
		}
		lazy[k] = 0;
	}
	void add(ll x, int a, int b, int k = 0, int l = 0, int r = -1) {
		if (r < 0)r = n;
		eval(k, l, r);
		if (r <= a || b <= l)return;
		if (a <= l && r <= b) {
			lazy[k] += x; eval(k, l, r);
		}
		else {
			add(x, a, b, k * 2 + 1, l, (l + r) / 2);
			add(x, a, b, k * 2 + 2, (l + r) / 2, r);
			node[k] = f(node[k * 2 + 1], node[k * 2 + 2]);
		}
	}
	ll query(int a, int b, int k = 0, int l = 0, int r = -1) {
		if (r < 0)r = n;
		eval(k, l, r);
		if (r <= a || b <= l)return init_c;
		if (a <= l && r <= b)return node[k];
		else {
			ll vl = query(a, b, k * 2 + 1, l, (l + r) / 2);
			ll vr = query(a, b, k * 2 + 2, (l + r) / 2, r);
			return f(vl, vr);
		}
	}
};

struct edge {
	int to;
};
using edges = vector<edge>;
using Graph = vector<edges>;
struct HLDecomposition {
	struct Chain {
		int depth;
		P parent;//chain number,index
		vector<P> child;//child chain number,parent index
		vector<int> mapfrom;
		SegT stree;
		SegT2 idst;

		Chain() { ; }
		Chain(int n) {
			stree.init(n), idst.init(n);
		}
	};
	Graph baseG;
	vector<Chain> chains;
	vector<P> mapto;//raw index->chain number &index
	vector<vector<int>> mapfrom;//chain number & index ->raw index

	HLDecomposition() { ; }
	HLDecomposition(const Graph &g) {
		baseG = g;
		const int n = baseG.size();
		mapto = vector<P>(n, P{ -1,-1 });
		mapfrom.clear();
		vector<int> sz(n, 0);
		int start = -1;
		rep(i, n)if (baseG[i].size() <= 1) { start = i; break; }
		//assert(start != -1);
		size_check_bfs(start, sz);
		decomposition(start, start, 0, 0, 0, sz);
	}
	int depth(int t) {
		return chains[mapto[t].first].depth;
	}

private:
	void size_check_bfs(int start, vector<int> &sz) {
		const int n = baseG.size();
		queue<P> que;
		que.push({ start,start });
		int cnt = 0; vector<int> ord(n, -1);
		while (!que.empty()) {
			int from, parent;
			tie(from, parent) = que.front(); que.pop();
			ord[cnt++] = from;
			for (edge e : baseG[from]) {
				if (e.to == parent)continue;
				que.push({ e.to,from });
			}
		}
		//assert(cnt == n);
		reverse(all(ord));
		rep(i, n) {
			int from = ord[i];
			sz[from] = 1; for (edge e : baseG[from])sz[from] += sz[e.to];
		}
	}
	int decomposition(int from, int parent, int depth, int pnumber, int pindex, const vector<int> &sz) {
		vector<int> seq;
		bfs(from, parent, seq, sz);
		const int c = chains.size();
		chains.push_back(Chain((int)seq.size()));
		//chains.push_back(Chain());
		chains[c].depth = depth;
		chains[c].parent = { pnumber,pindex };
		rep(i, seq.size()) {
			mapto[seq[i]] = { c,i };
			chains[c].mapfrom.push_back(seq[i]);
		}
		mapfrom.push_back(chains[c].mapfrom);
		rep(i, seq.size()) {
			for (edge e : baseG[seq[i]]) {
				if (mapto[e.to].first != -1)continue;
				int nc = decomposition(e.to, seq[i], depth + 1, c, i, sz);
				chains[c].child.push_back({ nc,i });
			}
		}
		return c;
	}
	void bfs(int from, int parent, vector<int> &seq, const vector<int> &sz) {
		for (;;) {
			seq.push_back(from);
			int best = -1, next = -1;
			for (edge e : baseG[from]) {
				if (e.to == parent)continue;
				if (best < sz[e.to]) {
					best = sz[e.to]; next = e.to;
				}
			}
			if (next == -1)break;
			parent = from; from = next;
		}
	}
	vector<pair<int, P>> all_edge(int u, int v) {
		vector<pair<int, P>> res;
		if (depth(u) > depth(v))swap(u, v);
		while (depth(v) > depth(u)) {
			res.push_back({ mapto[v].first,{ 0,mapto[v].second+1 } });
			P par = chains[mapto[v].first].parent;
			v = mapfrom[par.first][par.second];
		}
		while (mapto[v].first != mapto[u].first) {
			res.push_back({ mapto[v].first,{ 0,mapto[v].second+1 } });
			P par = chains[mapto[v].first].parent;
			v = mapfrom[par.first][par.second];
			res.push_back({ mapto[u].first,{ 0,mapto[u].second+1 } });
			par = chains[mapto[u].first].parent;
			u = mapfrom[par.first][par.second];
		}
		P p = minmax(mapto[v].second, mapto[u].second);
		res.push_back({ mapto[v].first,{ p.first+1,p.second+1 } });
		return res;
	}

public:

	int calc_id(int u, int v) {
		vector<pair<int, P>> es = all_edge(u, v);
		int mi = mod;
		int res = 0;
		rep(i, es.size()) {
			int id = es[i].first;
			int l = es[i].second.first; int r = es[i].second.second;
			//cout << id << " " << l << " " << r << "\n";
			
			int val = chains[id].stree.query(l, r);
			if (val < mi) {
				mi = val;
				res = chains[id].idst.query(l, r);
			}
		}
		assert(mi < mod);
		if (mi!=1)return -1;
		return res;
	}
	vector<P> sones(int u, int v) {
		vector<pair<int, P>> es = all_edge(u, v);
		vector<P> res;
		rep(i, es.size()) {
			int id = es[i].first;
			int l = es[i].second.first; int r = es[i].second.second;
			vector<int> v;
			chains[id].stree.onefind(v, l,r);
			for (int loc : v) {
				if (loc > 0) {
					res.push_back({ mapfrom[id][loc - 1],mapfrom[id][loc] });
				}
				else {
					int nw = mapfrom[id][0];
					int par = mapfrom[chains[id].parent.first][chains[id].parent.second];
					res.push_back({ par,nw });
				}
			}
		}
		return res;
	}
	void add(int u, int v, int x, int y) {
		vector<pair<int, P>> es = all_edge(u, v);
		rep(i, es.size()) {
			int id = es[i].first;
			int l = es[i].second.first; int r = es[i].second.second;
			chains[id].stree.add(x,l, r);
			chains[id].idst.add(y, l, r);
		}
	}
	vector<P> staones() {
		vector<P> res;
		rep(j,chains.size()) {
			Chain &c = chains[j];
			vector<int> v;
			c.stree.onefind(v,0, c.mapfrom.size());
			for (int id : v) {
				if (id > 0)res.push_back({ mapfrom[j][id - 1],mapfrom[j][id] });
				else {
					int nw = mapfrom[j][0];
					int par = mapfrom[chains[j].parent.first][chains[j].parent.second];
					res.push_back({ par,nw });
				}
			}
		}
		return res;
	}
};

void solve() {
	int n; cin >> n;
	Graph g(n);
	map<P, bool> exi;
	rep(i, n - 1) {
		int a, b; cin >> a >> b; a--; b--;
		g[a].push_back({ b });
		g[b].push_back({ a });
	}
	HLDecomposition hld(g);
	vector<int> c(n - 1), d(n - 1);
	rep(i, n - 1) {
		cin >> c[i] >> d[i]; c[i]--; d[i]--;
		hld.add(c[i], d[i], 1, i);
	}
	queue<P> q;
	vector<P> sones = hld.staones();
	for (P p : sones)q.push(p);
	int tmp = 0;

	while (!q.empty()) {
		tmp++;
		P p = q.front(); q.pop();
		int a = p.first, b = p.second;
		//cout << "hello "<<a << " " << b << "\n";
		int id = hld.calc_id(a, b);
		if (id < 0) {
			cout << "NO\n"; return;
		}
		hld.add(a, b, mod, 0);
		hld.add(c[id], d[id], -1, -id);
		vector<P> nex = hld.sones(c[id],d[id]);
		for (P p : nex)q.push(p);
	}
	if (tmp != n - 1) {
		cout << "NO\n"; return;
	}
	cout << "YES\n";
}



struct uf {
private:
	vector<int> par, ran;
public:
	uf(int n) {
		par.resize(n, 0);
		ran.resize(n, 0);
		rep(i, n) {
			par[i] = i;
		}
	}
	int find(int x) {
		if (par[x] == x)return x;
		else return par[x] = find(par[x]);
	}
	void unite(int x, int y) {
		x = find(x), y = find(y);
		if (x == y)return;
		if (ran[x] < ran[y]) {
			par[x] = y;
		}
		else {
			par[y] = x;
			if (ran[x] == ran[y])ran[x]++;
		}
	}
	bool same(int x, int y) {
		return find(x) == find(y);
	}
};
void generate_tree(int sz) {
	vector<pair<int, int>> res;
	random_device rnd;
	mt19937 mt(rnd());
	uf u(sz);
	for (int i = 0; i < sz - 1; i++) {
		int a = 0, b = 0;
		while (u.same(a, b)) {
			a = mt() % sz; b = mt() % sz;
			if (!u.same(a, b))break;
		}
		u.unite(a, b);
		res.push_back({ a,b });
	}
	rep(i, sz - 1) {
		cout << res[i].first+1 << " " << res[i].second+1 << "\n";
	}
}

signed main() {
	ios::sync_with_stdio(false);
	cin.tie(0);
	//cout << fixed << setprecision(10);
	//init_f();
	//init();
	//generate_tree(7);
	//generate_tree(7);
	solve();
	stop
		return 0;
}

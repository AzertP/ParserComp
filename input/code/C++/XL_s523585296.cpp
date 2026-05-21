#include <stdio.h>
#include <sstream>
#include <string.h>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <set>
#include <cctype>
#include <queue>
#include <stack>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <deque>
#include <limits>
#include <iomanip>
#include <ctype.h>
#include <unordered_map>
#include <random>
#include <numeric>
#include <iostream>
#include <array>

#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<ll, double> pld;
typedef pair<double, double> pdd;
typedef pair<double, ll> pdl;
typedef pair<int, char> pic;
typedef vector<ll> vl;
typedef vector<int> vi;
typedef priority_queue<ll, vector<ll>, greater<ll>> llgreaterq;
typedef priority_queue<pll, vector<pll>, greater<pll>> pllgreaterq;
typedef priority_queue<pair<ll, pll>, vector<pair<ll, pll>>, greater<pair<ll, pll>>> plpllgreaterq;
typedef priority_queue<vi, vector<vi>, greater<vi>> vigreaterq;
typedef priority_queue<vl, vector<vl>, greater<vl >> vlgreaterq;
template <class o,class p,class q>
using tuple3q = priority_queue<tuple<o, p, q>, vector<tuple<o, p, q>>, greater<tuple<o,p,q>>> ;
template <class o, class p, class q, class r>
using tuple4q = priority_queue<tuple<o, p, q,r>, vector<tuple<o, p, q,r>>, greater<tuple<o, p, q,r>>>;
int dx[] = { -1,0,1,0 };
int dy[] = { 0,-1,0,1 };
#define bit(x,v) ((ll)x << v)
#define rep(x,n) for(ll x = 0;x < n;x++)
#define rep2(x,f,v) for(ll x=f;x<v;x++)
#define repe(v,x) for(auto v : x)
// 許容する誤差ε
#define EPS (1e-10)
// 2つのスカラーが等しいかどうか
#define EQ(a,b) (std::abs(a-b) < EPS)
// 2つのベクトルが等しいかどうか
#define EQV(a,b) ( EQ((a).real(), (b).real()) && EQ((a).imag(), (b).imag()) )
#define all(a) a.begin(),a.end()
#define all0(a) memset(a,0,sizeof(a))
#define allm1(a) memset(a,-1,sizeof(a))
#define put_float(v) 	cout << fixed << setprecision(10); \
						cout << v << endl
#define vinsert(v,p,x) v.insert(v.begin() + p,x)
#define vsort(v) sort(all(v));
#define dup(v) v.erase(unique(all(v)),v.end())
#define ion(i,j) ((i & (1LL << j)) > 0)
#define next(i) i++;i%=2
#define Len size()
#define ull unsignd long long
#define psp(a,b) push_back(make_pair(a,b))
#define psp2(a,b) push(make_pair(a,b))
#define cini(a) a; cin >> a
#define infa(a,b) (a + b) % INF
#define infm(a,b) (a * b) % INF
#define infd(a,b) (a * modinv(b)) % INF
#define infs(a,b) (a + INF - b) % INF
#define inf(a) (a) %= INF
#define inff(a) ((a) % INF)
#define No cout << "No" << endl
#define Yes cout << "Yes" << endl
#define NO cout << "NO" << endl
#define YES cout << "YES" << endl
#define smal -INF*INF;
#define big INF*INF;
const ll INF = 1000000007;
const int MAX = 2000010;
const int MOD = 1000000007;

long long fac[MAX], finv[MAX], inv[MAX];
void COMinit() {
	fac[0] = fac[1] = 1;
	finv[0] = finv[1] = 1;
	inv[1] = 1;
	for (int i = 2; i < MAX; i++) {
		fac[i] = fac[i - 1] * i % MOD;
		inv[i] = MOD - inv[MOD % i] * (MOD / i) % MOD;
		finv[i] = finv[i - 1] * inv[i] % MOD;
	}
}

// 二項係数計算
long long COM(int n, int k) {
	if (n < k) return 0;
	if (n < 0 || k < 0) return 0;
	return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD;
}

ll getpow(ll b, ll x, ll md) {
	ll t = b;
	ll res = 1;
	while (x > 0)
	{
		if (x & 1) {
			res *= t;
			res %= md;
		}
		x >>= 1;
		t *= t;
		t %= md;
	}
	return res;
}
ll getpow(ll b, ll x) {

	return getpow(b, x, INF);
}
ll modinv(ll x) {
	return getpow(x, INF - 2);
}
ll gcd(ll a, ll b) {
	if (b == 0) return a;
	return gcd(b, a % b);
}
int pr[100010];
int lank[100010];
void uini(int n) {
	for (size_t i = 0; i <= n; i++)
	{
		pr[i] = i;
		lank[i] = 1;
	}
}

int parent(int x) {
	if (x == pr[x]) return x;
	return pr[x] = parent(pr[x]);
}

int same(int x, int y) {
	return parent(x) == parent(y);
}

bool unit(int x, int y) {

	int px = parent(x);
	int py = parent(y);

	if (px == py) return false;
	if (lank[py] < lank[px]) {
		pr[py] = px;
		lank[px] += lank[py];
	}
	else {
		pr[px] = py;
		lank[py] += lank[px];
	}
	return true;
}

ll merge(ll* a, int left, int mid, int right) {
	ll n1 = mid - left;
	ll n2 = right - mid;
	vector<int> L(n1 + 1);
	vector<int> R(n2 + 1);
	for (size_t i = 0; i < n1; i++)
	{
		L[i] = a[left + i];
	}
	for (size_t i = 0; i < n2; i++)
	{
		R[i] = a[mid + i];
	}

	L[n1] = INF;
	R[n2] = INF;
	ll i = 0;
	ll j = 0;
	ll r = 0;
	for (size_t k = left; k < right; k++)
	{
		if (L[i] <= R[j]) {
			a[k] = L[i];
			i++;
		}
		else {
			a[k] = R[j];
			r += n1 - i;
			j++;
		}
	}
	return r;
}
ll merge2(pair<int, char>* a, int left, int mid, int right) {
	ll n1 = mid - left;
	ll n2 = right - mid;
	vector<pair<int, char>> L(n1 + 1);
	vector<pair<int, char>> R(n2 + 1);
	for (size_t i = 0; i < n1; i++)
	{
		L[i] = a[left + i];
	}
	for (size_t i = 0; i < n2; i++)
	{
		R[i] = a[mid + i];
	}

	L[n1] = make_pair(INF, ' ');
	R[n2] = make_pair(INF, ' ');
	ll i = 0;
	ll j = 0;
	ll r = 0;
	for (size_t k = left; k < right; k++)
	{
		if (L[i].first <= R[j].first) {
			a[k] = L[i];
			i++;
		}
		else {
			a[k] = R[j];
			r += n1 - i;
			j++;
		}
	}
	return r;
}
ll mergeSort2(pair<int, char>* a, int left, int right) {
	ll res = 0;
	if (left + 1 < right) {
		int mid = (left + right) / 2;
		res = mergeSort2(a, left, mid);
		res += mergeSort2(a, mid, right);
		res += merge2(a, left, mid, right);
	}
	return res;
}
ll mergeSort(ll* a, int left, int right) {
	ll res = 0;
	if (left + 1 < right) {
		int mid = (left + right) / 2;
		res = mergeSort(a, left, mid);
		res += mergeSort(a, mid, right);
		res += merge(a, left, mid, right);
	}
	return res;
}
int partition(pair<int, char>* a, int p, int r) {
	pair<int, char> x = a[r];
	int i = p - 1;
	for (size_t j = p; j < r; j++)
	{
		if (a[j].first <= x.first) {
			i++;
			swap(a[i], a[j]);
		}
	}
	swap(a[i + 1], a[r]);
	return i + 1;
}
void quick(pair<int, char>* a, int p, int r) {
	if (p < r) {
		int q = partition(a, p, r);
		quick(a, p, q - 1);
		quick(a, q + 1, r);
	}
}

ll n;
int ci = 0;
ll P[1000010];
struct Node {
	int key;
	int priority;
	Node* parent, * left, * right;
	Node(int key, int priority);
	Node() {}
};
Node NIL;
Node::Node(int key, int priority) : key(key), priority(priority) {
	left = &NIL;
	right = &NIL;
}
Node* root = new Node();
void cenrec(Node* k) {
	if (k->key == NIL.key) return;
	cenrec(k->left);
	cout << " " << k->key;
	cenrec(k->right);
}
void fastrec(Node* k)
{
	if (k->key == NIL.key) return;
	cout << " " << k->key;
	fastrec(k->left);
	fastrec(k->right);
}
void insert(Node* v) {
	Node* y = &NIL;
	Node* x = root;
	while (x->key != NIL.key)
	{
		y = x;
		if (v->key < x->key) {
			x = x->left;
		}
		else {
			x = x->right;
		}
	}
	v->parent = y;
	if (y->key == NIL.key) {
		root = v;
	}
	else if (v->key < y->key) {
		y->left = v;
	}
	else {
		y->right = v;
	}

}

Node* find(Node* k, ll v)
{
	if (k->key == NIL.key) return &NIL;
	if (k->key == v) return k;
	if (v < k->key) return find(k->left, v);
	return find(k->right, v);
}
void delp12(Node* x) {
	if (x->key == NIL.key)  return;
	Node* l = x->left;
	Node* r = x->right;
	Node* pr = x->parent;

	if (l->key == NIL.key
		&& r->key == NIL.key) {
		if (pr->left == x) {
			pr->left = &NIL;
		}
		else pr->right = &NIL;
	}
	else if (l->key != NIL.key) {
		if (pr->left == x) {
			pr->left = l;
		}
		else pr->right = l;
		l->parent = pr;
	}
	else if (r->key != NIL.key) {
		if (pr->left == x) {
			pr->left = r;
		}
		else pr->right = r;
		r->parent = pr;
	}
}
Node* get_next(Node* k) {
	if (k->key == NIL.key) return &NIL;
	Node* res = get_next(k->left);
	if (res->key != NIL.key) return res;
	return k;
}
void del(Node* x) {

	if (x->key == NIL.key) return;
	Node* l = x->left;
	Node* r = x->right;
	Node* pr = x->parent;

	if (l->key != NIL.key && r->key != NIL.key) {
		Node* nex = get_next(r);
		x->key = nex->key;
		delp12(nex);
	}
	else {
		delp12(x);
	}
}
Node* rightRotate(Node* t) {
	Node* s = t->left;
	t->left = s->right;
	s->right = t;
	return s;
}
Node* leftRotate(Node* t) {
	Node* s = t->right;
	t->right = s->left;
	s->left = t;
	return s;
}
Node* _insert(Node* t, int key, int priority) {
	if (t->key == NIL.key) {
		return new Node(key, priority);
	}
	if (key == t->key) {
		return t;
	}

	if (key < t->key) {
		t->left = _insert(t->left, key, priority);
		if (t->priority < t->left->priority) {
			t = rightRotate(t);
		}
	}
	else {
		t->right = _insert(t->right, key, priority);
		if (t->priority < t->right->priority) {
			t = leftRotate(t);
		}
	}
	return t;
}
Node* delete1(Node* t, int key);
Node* _delete(Node* t, int key) {
	if (t->left->key == NIL.key && t->right->key == NIL.key) {
		return &NIL;
	}
	else if (t->left->key == NIL.key) {
		t = leftRotate(t);
	}
	else if (t->right->key == NIL.key) {
		t = rightRotate(t);
	}
	else
	{
		if (t->left->priority > t->right->priority) {
			t = rightRotate(t);
		}
		else
			t = leftRotate(t);
	}
	return delete1(t, key);
}
Node* delete1(Node* t, int key) {
	if (t->key == NIL.key) {
		return &NIL;
	}
	if (key < t->key) {
		t->left = delete1(t->left, key);
	}
	else if (key > t->key) {
		t->right = delete1(t->right, key);
	}
	else return _delete(t, key);
	return t;
}
int H;
int left(int i) {
	return i * 2 + 1;
}
int right(int i) {
	return i * 2 + 2;
}
struct edge {
	int from, to;
	ll val;
	edge(int from, int to, ll val) : from(from), to(to), val(val) {}
};
ll k;
int _rank[1010];
int temp[1010];
bool compare_sa(int i, int j) {
	if (_rank[i] != _rank[j]) return _rank[i] < _rank[j];
	else {
		int ri = i + k <= n ? _rank[i + k] : -1;
		int rj = j + k <= n ? _rank[j + k] : -1;
		return ri < rj;
	}
}
void construct_sa(string S, int* sa) {
	n = S.length();

	for (size_t i = 0; i <= n; i++)
	{
		sa[i] = i;
		_rank[i] = i < n ? S[i] : -1;
	}

	for (k = 1; k <= n; k *= 2)
	{
		sort(sa, sa + n + 1, compare_sa);

		// saはソート後の接尾辞の並びになっている、rankは元の位置のindexを保持したまま、更新されている。
		// ピンとこなかった部分
		temp[sa[0]] = 0;
		for (size_t i = 1; i <= n; i++)
		{
			temp[sa[i]] = temp[sa[i - 1]] + (compare_sa(sa[i - 1], sa[i]) ? 1 : 0);
		}
		for (size_t i = 0; i <= n; i++)
		{
			_rank[i] = temp[i];
		}
	}
}
bool contain(string S, int* sa, string T) {
	int a = 0, b = S.length();
	// sa は 接尾辞が辞書順に並んでいる、入っているのはその位置のインデックス
	while (b - a > 1) {
		int c = (a + b) / 2;
		if (S.compare(sa[c], T.length(), T) < 0) a = c;
		else b = c;
	}
	return S.compare(sa[b], T.length(), T) == 0;
}


#define bit(x,v) ((ll)x << v)

class BIT {

	static const int MAX_N = 500010;
public:
	BIT() { memset(bit, 0, sizeof(bit)); }
	ll bit[MAX_N + 1], n;
	ll sum(int i) {
		ll s = 0;
		while (i > 0)
		{
			s += bit[i];
			i -= i & -i;
		}
		return s;
	}

	void add(int i, int x) {
		while (i <= n)
		{
			bit[i] += x;
			i += i & -i;
		}
	}


};
struct UnionFind {
	vector<int> A;
	UnionFind(int n) : A(n, -1) {}
	int find(int x) {
		if (A[x] < 0) return x;
		return A[x] = find(A[x]);
	}
	void unite(int x, int y) {
		x = find(x), y = find(y);
		if (x == y) return;
		if (A[x] > A[y]) swap(x, y);
		A[x] += A[y];
		A[y] = x;
	}
	int ngroups() {
		int ans = 0;
		for (auto a : A) if (a < 0) ans++;
		return ans;
	}
};
void yes() { cout << "Yes\n"; exit(0); }
void no() { cout << "No\n"; exit(0); }
vector<ll> getp(ll n) {

	vector<ll> res;
	ll a = 2;
	if (n % 2 == 0) {
		res.push_back(2);
		while (n % 2 == 0)n /= 2;
	}

	for (ll i = 3; i * i <= n; i += 2)
	{
		if (n % i == 0) {
			res.push_back(i);
			while (n % i == 0)n /= i;
		}
	}
	if (n != 1) res.push_back(n);
	return res;
}
vector<ll> getp2(ll n) {

	vector<ll> res;
	ll a = 2;
	if (n % 2 == 0) {

		while (n % 2 == 0) { n /= 2; res.push_back(2); }
	}

	for (ll i = 3; i * i <= n; i += 2)
	{
		if (n % i == 0) {

			while (n % i == 0) { n /= i; res.push_back(i); }
		}
	}
	if (n != 1) res.push_back(n);
	return res;
}
vector<pll> getp3(ll n) {
	vector<pll> res;
	ll a = 2;
	int si = 0;
	if (n % 2 == 0) {

		res.push_back(make_pair(2, 0));
		while (n % 2 == 0) { n /= 2; res[si].second++; }
		si++;
	}

	for (ll i = 3; i * i <= n; i += 2)
	{
		if (n % i == 0) {
			res.push_back(make_pair(i, 0));
			while (n % i == 0) { n /= i; res[si].second++; }
			si++;
		}
	}
	if (n != 1) { res.push_back(make_pair(n, 1)); }
	return res;
}

vector<ll> getDivisors(ll n) {

	vector<ll> res;
	ll a = 2;
	res.push_back(1);
	for (ll i = 2; i * i <= n; i++)
	{

		if (n % i == 0) {
			res.push_back(i);
			if (n / i != i)
				res.push_back(n / i);
		}
	}
	return res;
}

struct ve {
public:
	vector<ve> child;
	int _t = INF;
	ve(int t) :_t(t) {}
	ve(ve _left, ve _right) {
		_t = _left._t + _right._t;
		child.push_back(_left);
		child.push_back(_right);
	}
	bool operator<(const ve& t) const {
		return _t > t._t;
	}
};

vector<bool> elas(ll n) {
	vector<bool> r(n);
	fill(r.begin(), r.end(), 1);
	r[0] = 0;
	r[1] = 0;
	for (ll i = 2; i * i < n; i++)
	{
		if (!r[i]) continue;
		ll ti = i * 2;
		while (ti < n)
		{
			r[ti] = false;
			ti += i;
		}
	}
	return r;
}
bool isPrime(ll v) {
	for (ll i = 2; i * i <= v; i++)
	{
		if (v % i == 0) return false;
	}
	return true;
}


class SegTree {

public:
	const static int MAX_N = 100010;
	const static int DAT_SIZE = (1 << 18) - 1;
	int N, Q;
	int A[MAX_N];


	ll data[DAT_SIZE], datb[DAT_SIZE];
	void init(int _n) {
		N = 1;
		while (N < _n) N <<= 1;
		memset(data, 0, sizeof(data));
		memset(datb, 0, sizeof(datb));
	}
	void init(int _n, ll iv) {
		N = 1;
		while (N < _n) N <<= 1;
		rep(i, DAT_SIZE) {
			data[i] = iv;
			datb[i] = iv;
		}
	}
	void add(int a, int b, int x) {
		add(a, b + 1, x, 0, 0, N);
	}
	void add(int a, int b, int x, int k, int l, int r) {
		if (a <= l && r <= b) {
			data[k] += x;
		}
		else if (l < b && a < r) {
			datb[k] += (min(b, r) - max(a, l)) * x;
			add(a, b, x, k * 2 + 1, l, (l + r) / 2);
			add(a, b, x, k * 2 + 2, (l + r) / 2, r);
		}
	}

	void change(int a, int b, int x) {
		change(a, b + 1, x, 0, 0, N);
	}
	void change(int a, int b, int x, int k, int l, int r) {
		if (a <= l && r <= b) {
			data[k] = x;
		}
		else if (l < b && a < r) {
			datb[k] = x;
			change(a, b, x, k * 2 + 1, l, (l + r) / 2);
			change(a, b, x, k * 2 + 2, (l + r) / 2, r);
		}
	}

	ll sum(int a, int b) {
		return sum(a, b + 1, 0, 0, N);
	}
	ll sum(int a, int b, int k, int l, int r) {
		if (b <= l || r <= a) {
			return 0;
		}
		if (a <= l && r <= b) {
			return data[k] * (r - l) + datb[k];
		}

		ll res = (min(b, r) - max(a, l)) * data[k];
		res += sum(a, b, k * 2 + 1, l, (l + r) / 2);
		res += sum(a, b, k * 2 + 2, (l + r) / 2, r);
		return res;
	}
};
class Segment;
class Circle;

class Point {
public:
	double x, y;

	Point(double x = 0, double y = 0) :x(x), y(y) {}

	Point operator + (Point p) { return Point(x + p.x, y + p.y); }
	Point operator - (Point p) { return Point(x - p.x, y - p.y); }
	Point operator * (double a) { return Point(a * x, a * y); }
	Point operator / (double a) { return Point(x / a, y / a); }

	double abs() { return sqrt(norm()); }
	double norm() { return x * x + y * y; }

	bool operator < (const Point& p)const {
		return x != p.x ? x < p.x : y < p.y;
	}
	bool operator == (const Point& p) const {
		return fabs(x - p.x) < EPS && fabs(y - p.y) < EPS;
	}
	static double dot(Point a, Point b) {
		return a.x * b.x + a.y * b.y;
	}
	static double cross(Point a, Point b) {
		return a.x * b.y - a.y * b.x;
	}
	static bool isOrthogonal(Point a, Point b) {
		return EQ(dot(a, b), 0.0);
	}
	static bool isOrthogonal(Point a1, Point a2, Point b1, Point b2) {
		return isOrthogonal(a1 - a2, b1 - b2);
	}
	static bool isOrthogonal(Segment s1, Segment s2);

	static bool isPalallel(Point a, Point b) {
		return EQ(cross(a, b), 0.0);
	}
	static bool isPalallel(Point a1, Point a2, Point b1, Point b2) {
		return isPalallel(a1 - a2, b1 - b2);
	}
	static bool isPalallel(Segment s1, Segment s2);

	static const int COUNTER_CLOCKWISE = 1;
	static const int CLOCKWISE = -1;
	static const int ONLINE_BACK = 2;
	static const int ONLINE_FRONT = -2;
	static const int ON_SEGMENT = 0;
	static int ccw(Point p0, Point p1, Point p2) {
		// 線分はp0とp1でp2がどこにあるかを探る
		Point a = p1 - p0;
		Point b = p2 - p0;
		if (cross(a, b) > EPS) return COUNTER_CLOCKWISE;
		if (cross(a, b) < -EPS) return CLOCKWISE;
		if (dot(a, b) < -EPS) return ONLINE_BACK;
		if (a.norm() < b.norm()) return ONLINE_FRONT;
		return ON_SEGMENT;
	}

	static bool intersect(Point p1, Point p2, Point p3, Point p4) {
		return (ccw(p1, p2, p3) * ccw(p1, p2, p4) <= 0
			&& ccw(p3, p4, p1) * ccw(p3, p4, p2) <= 0);
	}
	static bool intersect(Segment s1, Segment s2);
	static Point project(Segment s, Point p);

	static Point reflect(Segment s, Point p);

	static Point getDistance(Point a, Point b) {
		return (a - b).abs();
	}

	static double getDistanceLP(Segment s, Point p);

	static double getDistanceSP(Segment s, Point p);

	static double getDistance(Segment s1, Segment s2);

	static Point getIntersection(Segment s1, Segment s2);

	static pair<Point, Point> crossPoints(Circle c, Segment s);

	static int contains(vector<Point> g, Point p) {
		int n = g.size();
		bool x = false;
		rep(i, n) {
			Point a = g[i] - p, b = g[(i + 1) % n] - p;
			// 線の上に載っているか
			if (std::abs(cross(a, b)) < EPS && dot(a, b) < EPS) return 1;

			// pを基準として上下にあるか
			// または外積が正か?(→にあるか)
			if (a.y > b.y) swap(a, b);
			if (a.y < EPS && EPS < b.y && cross(a, b) > EPS) x = !x;
		}
		return x ? 2 : 0;
	}

	static vector<Point> andrewScan(vector<Point> s) {
		vector<Point> u, l;
		if (s.size() < 3) return s;
		sort(all(s));
		u.push_back(s[0]);
		u.push_back(s[1]);
		l.push_back(s[s.size() - 1]);
		l.push_back(s[s.size() - 2]);

		for (int i = 2; i < s.size(); i++) {

			for (int _n = u.size(); _n >= 2 && ccw(u[_n - 2], u[_n - 1], s[i]) > CLOCKWISE; _n--) {
				u.pop_back();
			}
			u.push_back(s[i]);
		}

		for (int i = s.size() - 3; i >= 0; i--) {

			for (int _n = l.size(); _n >= 2 && ccw(l[_n - 2], l[_n - 1], s[i]) > CLOCKWISE; _n--) {
				l.pop_back();
			}
			l.push_back(s[i]);
		}

		reverse(all(l));
		for (int i = u.size() - 2; i >= 1; i--)
		{
			l.push_back(u[i]);
		}

		return l;
	}
	void get_cin() {
		cin >> x >> y;
	}
};

class Segment {
public:
	Point p1, p2;
	Segment() {}
	Segment(Point p1, Point p2) :p1(p1), p2(p2) {}
	void get_cin() {
		cin >> p1.x >> p1.y >> p2.x >> p2.y;
	}
	Point p1tp2() {
		return p2 - p1;
	}
	Point p2tp1() {
		return p1 - p2;
	}
	double abs() {
		return std::abs(norm());
	}
	double norm() {
		return (p2 - p1).norm();
	}
};

bool Point::isOrthogonal(Segment s1, Segment s2) {
	return EQ(dot(s1.p2 - s1.p1, s2.p2 - s2.p1), 0.0);
}
bool Point::isPalallel(Segment s1, Segment s2) {
	return EQ(cross(s1.p2 - s1.p1, s2.p2 - s2.p1), 0.0);
}
bool Point::intersect(Segment s1, Segment s2) {
	return intersect(s1.p1, s1.p2, s2.p1, s2.p2);
}
Point Point::project(Segment s, Point p) {
	Point base = s.p2 - s.p1;
	double r = Point::dot(p - s.p1, base) / base.norm();
	return s.p1 + base * r;
}
Point Point::reflect(Segment s, Point p) {
	return (project(s, p) * 2) - p;
}
double Point::getDistanceLP(Segment s, Point p) {
	return std::abs(cross(s.p2 - s.p1, p - s.p1) / (s.p2 - s.p1).abs());
}
double Point::getDistanceSP(Segment s, Point p) {
	if (dot(s.p2 - s.p1, p - s.p1) < 0.0) return (p - s.p1).abs();
	if (dot(s.p1 - s.p2, p - s.p2) < 0.0) return (p - s.p2).abs();
	return getDistanceLP(s, p);
}
double Point::getDistance(Segment s1, Segment s2) {
	if (intersect(s1, s2)) return 0.0;
	return min({ getDistanceSP(s1,s2.p1),getDistanceSP(s1,s2.p2)
		,getDistanceSP(s2,s1.p1),getDistanceSP(s2,s1.p2) });
}

Point Point::getIntersection(Segment s1, Segment s2) {
	// (s1.p1 - s2.p1).norm()
	auto bs = s1.p2 - s1.p1;
	auto n1 = s2.p1 - s1.p1;
	auto n2 = s2.p2 - s1.p1;
	auto c1 = std::abs(cross(n1, bs)) / bs.norm();
	auto c2 = std::abs(cross(n2, bs)) / bs.norm();
	return s2.p1 + (s2.p2 - s2.p1) * (c1 / (c1 + c2));
	// c1:c2=t:1-t
	// c2t=(1-t)c1
	// t/(1-t)=c1/(c1+c2)
	// 
}

double arg(Point p) { return atan2(p.y, p.x); }
Point polar(double a, double r) { return Point(cos(r) * a, sin(r) * a); }
class Circle {
public:
	Point c;
	double r;
	Circle(Point c = Point(), double r = 0.0) : c(c), r(r) {}
	void get_cin() {
		cin >> c.x >> c.y >> r;
	}
	static pair<Point, Point> getCrossPoints(Circle c1, Circle c2) {
		double d = (c1.c - c2.c).abs(); // 中心点どうしの距離
		double a = acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2 * c1.r * d));
		double t = arg(c2.c - c1.c);
		return make_pair(c1.c + polar(c1.r, t + a), c1.c + polar(c1.r, t - a));

	}
};

pair<Point, Point> Point::crossPoints(Circle c, Segment s) {
	auto pp = project(s, c.c);
	auto f = (pp - c.c).norm();
	auto mu = sqrt(c.r * c.r - f);
	auto e = s.p1tp2() / s.p1tp2().abs();
	return make_pair(pp + e * mu, pp - e * mu);

}

ll divRm(string s, ll x) {

	ll r = 0;
	for (ll i = 0; i < s.size(); i++)
	{
		r *= 10;
		r += s[i] - '0';
		r %= x;
	}
	return r;
}
ll cmbi(ll x, ll b) {

	ll res = 1;
	for (size_t i = 0; i < b; i++)
	{
		res *= x - i;
		res %= INF;
		res *= inv[b - i];
		res %= INF;
	}
	return res;
}

double digsum(ll x) {
	ll res = 0;
	while (x > 0)
	{
		res += x % 10;
		x /= 10;
	}
	return res;
}
bool check_parindrome(string s) {
	int n = s.size();
	rep(i, n / 2) {
		if (s[i] != s[n - i - 1]) {
			return false;
		}
	}
	return true;
}


void solv() {
		
	string s,t;
	cin >> s >> t;
	for (char a = 'a'; a <= 'z'; a++) {
		if(s + a == t){
			Yes;
			return;
		}
	}
	No;
	
}


int main() {
	COMinit();
	solv();
	return 0;
}

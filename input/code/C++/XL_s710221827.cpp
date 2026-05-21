/*temp2*/
#define use_sort
#define use_sort
//
//
//
//

// //をuse_ooに置き換える
//#undef _DEBUG
//#pragma GCC optimize("Ofast")
//不動小数点の計算高速化
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#ifdef _DEBUG
#include "bits_stdc++.h"
#else
#include <bits/stdc++.h>
#endif

//todo use_pbdsを消すとバグる 後で直す
#define use_pbds
#ifdef use_pbds
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
template<class T, class U, class W, class X> auto count(__gnu_pbds::gp_hash_table<T, U, W> &a, X k) { return a.find(k) != a.end(); }
#endif

using namespace std;
using namespace std::chrono;

/*@formatter:off*/
#ifndef _DEBUG
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
namespace cmp = boost::multiprecision;
using cint = cmp :: cpp_int;
using bdou = cmp::cpp_dec_float_100;
#endif
#define ll long long
using sig_dou = double;

template<class T,class U> auto max(T a, U b){return a>b ? a: b;}
template<class T,class U> auto min(T a, U b){return a<b ? a: b;}

namespace{//helper
    template<class T>T s_decl2(const T& A){return A;}template<class T>auto s_decl2(const vector<T>& A){return s_decl2(A[0]);}
}
//vector<T>でTを返す
#define decl_t(A) decltype(A)::value_type

//vector<vector<.....T>>でTを返す
#define decl2(A) decltype(s_decl2(A))


#define int long long //todo 消したら動かない intの代わりにsignedを使う
auto start_time = system_clock::now();
auto past_time = system_clock::now();

#define debugName(VariableName) # VariableName
//最大引数がN
#define over2(o1, o2, name, ...) name
#define over3(o1, o2, o3, name, ...) name
#define over4(o1, o2, o3, o4, name, ...) name
#define over5(o1, o2, o3, o4, o5, name, ...) name
#define over6(o1, o2, o3, o4, o5, o6, name, ...) name
#define over7(o1, o2, o3, o4, o5, o6, o7, name, ...) name
#define over8(o1, o2, o3, o4, o5, o6, o7, o8, name, ...) name
#define over9(o1, o2, o3, o4, o5, o6, o7, o8, o9, name, ...) name
#define over10(o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, name, ...) name






#ifdef _DEBUG
string message;
string res_mes;
//#define use_debtor

#ifdef use_debtor
//https://marycore.jp/prog/cpp/class-extension-methods/ 違うかも
template<class T, class A = std::allocator<T>> struct debtor : std::vector<T, A> {
    using std::vector<T, A>::vector;
    template<class U> int deb_v(U a, int v) { return v; }
    template<class U> int deb_v(debtor<U> &a, int v = 0) {        cerr << a.size() << " ";        return deb_v(a.at(0), v + 1);    }
    template<class U> void deb_o(U a) { cerr << a << " "; }
    template<class U> void deb_o(debtor<U> &a) {        for (int i = 0; i < min((int) a.size(), 15ll); i++) { deb_o(a[i]); }        if ((int) a.size() > 15) { cerr << "..."; }        cerr << endl;    }
    typename std::vector<T>::reference my_at(typename std::vector<T>::size_type n, vector<int> &ind) {        if (n < 0 || n >= (int) this->size()) {            int siz = (int) this->size();            cerr << "vector size = ";            int dim = deb_v((*this));            cerr << endl;            ind.push_back(n);            cerr << "out index at  ";            for (auto &&i: ind) {                cerr << i << " ";            }            cerr << endl;            cerr << endl;            if (dim <= 2) { deb_o((*this)); }            exit(0);        }        return this->at(n);    }
    typename std::vector<T>::reference operator[](typename std::vector<T>::size_type n) {        if (n < 0 || n >= (int) this->size()) {            int siz = (int) this->size();            cerr << "vector size = ";            int dim = deb_v((*this));            cerr << endl;            cerr << "out index at  " << n << endl;            cerr << endl;            if (dim <= 2) { deb_o((*this)); }            exit(0);        }        return this->at(n);    }
};
#define vector debtor
#endif
#ifdef use_pbds
template<class T> struct my_pbds_tree {    set<T> s;    auto begin() { return s.begin(); }    auto end() { return s.end(); }    auto rbegin() { return s.rbegin(); }    auto rend() { return s.rend(); }    auto empty() { return s.empty(); }    auto size() { return s.size(); }    void clear() { s.clear(); }    template<class U> void insert(U v) { s.insert(v); }template<class U> void operator+=(U v) { insert(v); }    template<class F> auto erase(F v) { return s.erase(v); }    template<class U> auto find(U v) { return s.find(v); }    template<class U> auto lower_bound(U v) { return s.lower_bound(v); }    template<class U> auto upper_bound(U v) { return s.upper_bound(v); }    auto find_by_order(ll k) {        auto it = s.begin();        for (ll i = 0; i < k; i++)it++;        return it;    }    auto order_of_key(ll v) {        auto it = s.begin();        ll i=0;        for (;it != s.end() && *it <v ; i++)it++;        return i;    }};
#define pbds(T) my_pbds_tree<T>
#endif

//区間削除は出来ない
//gp_hash_tableでcountを使えないようにするため
template<class T, class U> struct my_unordered_map { unordered_map<T, U> m; my_unordered_map() {}; auto begin() { return m.begin(); } auto end() { return m.end(); } auto cbegin() { return m.cbegin(); } auto cend() { return m.cend(); } template<class V> auto erase(V v) { return m.erase(v); } void clear() { m.clear(); }    /*countは gp_hash_tableに存在しない*/    /*!= m.end()*/    template<class V> auto find(V v) { return m.find(v); } template<class V> auto &operator[](V n) { return m[n]; }};
#define unordered_map my_unordered_map
#define umapi unordered_map<ll,ll>
#define umapp unordered_map<P,ll>
#define umapu unordered_map<uint64_t,ll>
#define umapip unordered_map<ll,P>
template<class T, class U, class X> auto count(unordered_map<T, U> &a, X k) { return a.find(k) != a.end(); }
#else
#define endl '\n'
//umapはunorderd_mapになる
//umapiはgp_hash_table

//find_by_order(k) k番目のイテレーター
//order_of_key(k)  k以上が前から何番目か
#define pbds(U) __gnu_pbds::tree<U, __gnu_pbds::null_type, less<U>, __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update>

#define umapi __gnu_pbds::gp_hash_table<ll,ll,xorshift>
#define umapp __gnu_pbds::gp_hash_table<P,ll,xorshift>
#define umapu __gnu_pbds::gp_hash_table<uint64_t,ll,xorshift>
#define umapip __gnu_pbds::gp_hash_table<ll,P,xorshift>
#endif


struct xorshift {
    static uint64_t splitmix64(uint64_t x) {        x += 0x9e3779b97f4a7c15;        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;        return x ^ (x >> 31);    }
    size_t operator()(uint64_t x) const {        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();        return splitmix64(x + FIXED_RANDOM);    }
    size_t operator()(std::pair<ll, ll> x) const {        ll v = ((x.first) << 32) | x.second;        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();        return splitmix64(v + FIXED_RANDOM);    }
};
#ifdef use_pbds
template<class U, class L> void operator+=(__gnu_pbds::tree<U, __gnu_pbds::null_type, less<U>, __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update> &s, L v) { s.insert(v); }
#endif
//衝突対策
#define ws ws_

template<class A, class B, class C> struct T2 { A f;B s;C t;T2() { f = 0, s = 0, t = 0; }T2(A f, B s, C t) : f(f), s(s), t(t) {}bool operator<(const T2 &r) const { return f != r.f ? f < r.f : s != r.s ? s < r.s : t < r.t;        /*return f != r.f ? f > r.f : s != r.s ?n s > r.s : t > r.t; 大きい順 */   } bool operator>(const T2 &r) const { return f != r.f ? f > r.f : s != r.s ? s > r.s : t > r.t;        /*return f != r.f ? f > r.f : s != r.s ? s > r.s : t > r.t; 小さい順 */   } bool operator==(const T2 &r) const { return f == r.f && s == r.s && t == r.t; } bool operator!=(const T2 &r) const { return f != r.f || s != r.s || t != r.t; }};
template<class A, class B, class C, class D> struct F2 {    A a;    B b;    C c;    D d;    F2() { a = 0, b = 0, c = 0, d = 0; }    F2(A a, B b, C c, D d) : a(a), b(b), c(c), d(d) {}    bool operator<(const F2 &r) const { return a != r.a ? a < r.a : b != r.b ? b < r.b : c != r.c ? c < r.c : d < r.d;    /*    return a != r.a ? a > r.a : b != r.b ? b > r.b : c != r.c ? c > r.c : d > r.d;*/    }    bool operator>(const F2 &r) const { return a != r.a ? a > r.a : b != r.b ? b > r.b : c != r.c ? c > r.c : d > r.d;/*        return a != r.a ? a < r.a : b != r.b ? b < r.b : c != r.c ? c < r.c : d < r.d;*/    }    bool operator==(const F2 &r) const { return a == r.a && b == r.b && c == r.c && d == r.d; }    bool operator!=(const F2 &r) const { return a != r.a || b != r.b || c != r.c || d != r.d; }    ll operator[](ll i) {        assert(i < 4);        return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;    }};
typedef T2<ll, ll, ll> T;
typedef F2<ll, ll, ll, ll> F;
T mt(ll a, ll b, ll c) { return T(a, b, c); }
F mf(ll a, ll b, ll c, ll d) { return F(a, b, c, d); }


template<class T, class U> bool chma(T &a, const U &b) {    if (a < b) {        a = b;        return true;    }    return false;}
template<class T, class U> bool chmi(T &a, const U &b) {    if (b < a) {        a = b;        return true;    }    return false;}

//関数内をまとめる
//初期値l=0, r=-1
void pr_set_lr(int &l, int &r, int n) {    /*r==-1*/    if (!(~r)) {        if(!(~l)){            l = 0;            r = n;        }else {            r = l;            l = 0;        }    }}
//@マクロ省略系 型,構造
//using で元のdoubleを同時に使えるはず
#define double_big

#ifdef double_big
#define double long double
#define pow powl
#endif
using dou = double;
template<class T> T MAX() { return numeric_limits<T>::max(); }
template<class T> T MIN() { return numeric_limits<T>::min(); }
constexpr ll inf = (ll) 1e9 + 100;
constexpr ll linf = (ll) 1e18 + 100;
constexpr dou dinf = (dou) linf * linf;
constexpr char infc = '{';
const string infs = "{";

template<class T> T INF(){return MAX<T>() / 2;}
template<> signed INF(){return inf;}
template<> ll INF(){return linf;}
template<> double INF(){return dinf;}
template<> char INF(){return infc;}
template<> string INF(){return infs;}

const double eps = 1e-9;

//#define use_epsdou
#ifdef use_epsdou
//基本コメントアウト
struct epsdou {    double v;    epsdou(double v = 0) : v(v) {}    template<class T> epsdou &operator+=(T b) {        v += (double) b;        return (*this);    }    template<class T> epsdou &operator-=(T b) {        v -= (double) b;        return (*this);    }    template<class T> epsdou &operator*=(T b) {        v *= (double) b;        return (*this);    }    template<class T> epsdou &operator/=(T b) {        v /= (double) b;        return (*this);    }    epsdou operator+(epsdou b) { return v + (double) b; }    epsdou operator-(epsdou b) { return v - (double) b; }    epsdou operator*(epsdou b) { return v * (double) b; }    epsdou operator/(epsdou b) { return v / (double) b; }    epsdou operator-() const { return epsdou(-v); }    template<class T> bool operator<(T b) { return v < (double) b; }    template<class T> bool operator>(T b) {auto r = (double)b;        return v > (double) b; }    template<class T> bool operator==(T b) { return fabs(v - (double) b) <= eps; }    template<class T> bool operator<=(T b) { return v < (double) b || fabs(v - b) <= eps; }    template<class T> bool operator>=(T b) { return v > (double) b || fabs(v - b) <= eps; }    operator double() { return v; }};
template<>epsdou MAX(){return MAX<double>();}
template<>epsdou MIN(){return MIN<double>();}
//priqrity_queue等で使うのに必要
bool operator<(const epsdou &a, const epsdou &b) {return a.v < b.v;}
bool operator>(const epsdou &a, const epsdou &b) {return a.v > b.v;}
istream &operator>>(istream &iss, epsdou &a) {iss >> a.v;return iss;}
ostream &operator<<(ostream &os, epsdou &a) {os << a.v;return os;}
#define eps_conr_t(o) template<class T> epsdou operator o(T a, epsdou b) {return (dou) a o b.v;}
#define eps_conl_t(o) template<class T> epsdou operator o(epsdou a, T b) {return a.v o (dou) b;}
eps_conl_t(+)eps_conl_t(-)eps_conl_t(*)eps_conl_t(/)eps_conr_t(+)eps_conr_t(-)eps_conr_t(*)eps_conr_t(/)
//template<class U> epsdou max(epsdou a, U b){return a.v>b ? a.v: b;}
//template<class U> epsdou max(U a, epsdou b){return a>b.v ? a: b.v;}
//template<class U> epsdou min(epsdou a, U b){return a.v<b ? a.v: b;}
//template<class U> epsdou min(U a, epsdou b){return a<b.v ? a: b.v;}
#undef double
#define double epsdou
#undef dou
#define dou epsdou
#endif


#define ull unsigned long long
using itn = int;
using str = string;
using bo= bool;
#define au auto
using P = pair<ll, ll>;

#define fi first
#define se second
#define beg begin
#define rbeg rbegin
#define con continue
#define bre break
#define brk break
#define is ==
#define el else
#define elf else if
#define upd update
#define sstream stringstream

#define maxq 1
#define minq -1

#define ZERO(a) memset(a,0,sizeof(a))
#define MINUS(a) memset(a,0xff,sizeof(a))
#define MALLOC(type, len) (type*)malloc((len) * sizeof(type))
#define lam1(ret) [&](auto& v){return ret;}
#define lam2(v, ret) [&](auto& v){return ret;}
#define lam(...) over2(__VA_ARGS__,lam2,lam1)(__VA_ARGS__)
#define lamr(right) [&](auto& p){return p right;}
#define unique(v) v.erase( unique(v.begin(), v.end()), v.end() );
template<class T> bool distinct(const vector<T>& A){    auto B = A;    sort(B.begin(), B.end());    int N=(B.size());    unique(B);    return N == (int)(B.size());}
template<class H, class... T> bool distinct(vector<H> &A, H &head) {    A.push_back(head);    return distinct(A);}
template<class H, class... T>bool distinct(vector<H>& A, H& head,T&... a){    A.push_back(head);    return distinct(A, a...);}
template<class H, class... T>bool distinct(H& head,T&... a){    vector<H> A;    A.push_back(head);    return distinct(A, a...);}

//マクロ省略系 コンテナ
using vi = vector<ll>;
using vb = vector<bool>;
using vs = vector<string>;
using vd = vector<double>;
using vc = vector<char>;
using vp = vector<P>;
using vt = vector<T>;

//#define V vector
#define vvt0(t) vector<vector<t>>
#define vvt1(t, a) vector<vector<t>>a
#define vvt2(t, a, b) vector<vector<t>>a(b)
#define vvt3(t, a, b, c) vector<vector<t>> a(b,vector<t>(c))
#define vvt4(t, a, b, c, d) vector<vector<t>> a(b,vector<t>(c,d))

#define vvi(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(ll,__VA_ARGS__)
#define vvb(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(bool,__VA_ARGS__)
#define vvs(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(string,__VA_ARGS__)
#define vvd(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(double,__VA_ARGS__)
#define vvc(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(char,__VA_ARGS__)
#define vvp(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(P,__VA_ARGS__)
#define vvt(...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(T,__VA_ARGS__)
#define vv(type, ...) over4(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(type,__VA_ARGS__)

template<typename T> vector<T> make_v(size_t a) { return vector<T>(a); }
template<typename T, typename... Ts> auto make_v(size_t a, Ts... ts) { return vector<decltype(make_v<T>(ts...))>(a, make_v<T>(ts...)); }
#define vni(name, ...) auto name = make_v<ll>(__VA_ARGS__)
#define vnb(name, ...) auto name = make_v<bool>(__VA_ARGS__)
#define vns(name, ...) auto name = make_v<string>(__VA_ARGS__)
#define vnd(name, ...) auto name = make_v<double>(__VA_ARGS__)
#define vnc(name, ...) auto name = make_v<char>(__VA_ARGS__)
#define vnp(name, ...) auto name = make_v<P>(__VA_ARGS__)
#define vn(type, name, ...) auto name = make_v<type>(__VA_ARGS__)

#define PQ priority_queue<ll, vector<ll>, greater<ll> >
#define tos to_string
using mapi = map<ll, ll>;
using mapp = map<P, ll>;
using mapd = map<dou, ll>;
using mapc = map<char, ll>;
using maps = map<str, ll>;
using seti = set<ll>;
using setp = set<P>;
using setd = set<dou>;
using setc = set<char>;
using sets = set<str>;
using qui = queue<ll>;
#define uset unordered_set
#define useti unordered_set<ll,xorshift>
#define mset multiset
#define mseti multiset<ll>
#define umap unordered_map
#define mmap multimap
//任意のマクロサポート用 使う度に初期化する
int index_, v1_, v2_, v3_;

//マクロ省略形 関数等
#define arsz(a) (sizeof(a)/sizeof(a[0]))
#define sz(a) ((ll)(a).size())
#define mp make_pair
#define pb pop_back
#define pf push_front
#define eb emplace_back
#define all(a) (a).begin(),(a).end()
#define rall(a) (a).rbegin(),(a).rend()

string to_string(char c){string ret = "";ret+=c;return ret;}
template<class T> class pq_min_max {    vector<T> d;    void make_heap() {        for (int i = d.size(); i--;) {            if (i & 1 && d[i - 1] < d[i]) swap(d[i - 1], d[i]);            int k = down(i);            up(k, i);        }    }    inline int parent(int k) const { return ((k >> 1) - 1) & ~1; }    int down(int k) {        int n = d.size();        if (k & 1) { /* min heap*/            while (2 * k + 1 < n) {                int c = 2 * k + 3;                if (n <= c || d[c - 2] < d[c]) c -= 2;                if (c < n && d[c] < d[k]) {                    swap(d[k], d[c]);                    k = c;                }                else break;            }        }        else { /* max heap*/            while (2 * k + 2 < n) {                int c = 2 * k + 4;                if (n <= c || d[c] < d[c - 2]) c -= 2;                if (c < n && d[k] < d[c]) {                    swap(d[k], d[c]);                    k = c;                }                else break;            }        }        return k;    }    int up(int k, int root = 1) {        if ((k | 1) < (int) d.size() && d[k & ~1] < d[k | 1]) {            swap(d[k & ~1], d[k | 1]);            k ^= 1;        }        int p;        while (root < k && d[p = parent(k)] < d[k]) { /*max heap*/            swap(d[p], d[k]);            k = p;        }        while (root < k && d[k] < d[p = parent(k) | 1]) { /* min heap*/            swap(d[p], d[k]);            k = p;        }        return k;    }
public:
    pq_min_max() {}
    pq_min_max(const vector<T> &d_) : d(d_) { make_heap(); }
    template<class Iter> pq_min_max(Iter first, Iter last) : d(first, last) { make_heap(); }
    void operator+=(const T &x) {        int k = d.size();        d.push_back(x);        up(k);    }
    void pop_min() {        if (d.size() < 3u) { d.pop_back(); }        else {            swap(d[1], d.back());            d.pop_back();            int k = down(1);            up(k);        }    }
    void pop_max() {        if (d.size() < 2u) { d.pop_back(); }        else {            swap(d[0], d.back());            d.pop_back();            int k = down(0);            up(k);        }    }
    const T &get_min() const { return d.size() < 2u ? d[0] : d[1]; }
    const T &get_max() const { return d[0]; }
    int size() const { return d.size(); }
    bool empty() const { return d.empty(); }
};
//小さいほうからM個取得するpq
template<class T> struct helper_pq_size {    pq_min_max<T> q;    T su = 0;    int max_size = 0;    helper_pq_size(){}    helper_pq_size(int max_size) : max_size(max_size) {}    void clear() {        q = pq_min_max<T>();        su = 0;    }    void operator+=(T v) {        su += v;        q+=(v);        if (sz(q) > max_size){           su -= q.get_max();           q.pop_max();        }    }    T sum() { return su; }    T top() { return q.get_min(); }    void pop() {        su -= q.get_min();        q.pop_min();    }    T poll() {        T ret = q.get_min();        su -= ret;        q.pop_min();        return ret;    }    ll size() { return q.size(); }};
//大きいほうからM個取得するpq
template<class T> struct helper_pqg_size {    pq_min_max<T> q;    T su = 0;    int max_size = 0;    helper_pqg_size(){}    helper_pqg_size(int max_size) : max_size(max_size) {}    void clear() {        q = pq_min_max<T>();        su = 0;    }    void operator+=(T v) {        su += v;        q+=(v);        if (sz(q) > max_size){            su -= q.get_min();            q.pop_min();        }    }    T sum() { return su; }    T top() { return q.get_max(); }    void pop() {        su -= q.get_max();        q.pop_max();    }    T poll() {        T ret = q.get_max();        su -= ret;        q.pop_min();        return ret;    }    ll size() { return q.size(); }};;
template<class T> struct helper_pq {    priority_queue<T, vector<T>, greater<T> > q;/*小さい順*/    T su = 0;    helper_pq(){}    void clear() {        q = priority_queue<T, vector<T>, greater<T> >();        su = 0;    }    void operator+=(T v) {        su += v;        q.push(v);            }    T sum() { return su; }    T top() { return q.top(); }    void pop() {        su -= q.top();        q.pop();    }    T poll() {        T ret = q.top();        su -= ret;        q.pop();        return ret;    }    ll size() { return q.size(); }};
template<class T> struct helper_pqg {    priority_queue<T> q;/*大きい順*/    T su = 0;    helper_pqg(){}    void clear() {        q = priority_queue<T>();        su = 0;    }    void operator+=(T v) {        su += v;        q.push(v);            }    T sum() { return su; }    T top() { return q.top(); }    void pop() {su -= q.top();q.pop();}    T poll() {        T ret = q.top();        su -= ret;        q.pop();        return ret;    }    ll size() { return q.size(); }};

//小さいほうからsize個残る
template<class T> struct pq {
    helper_pq<T> a_q;/*大きい順*/
    helper_pq_size<T> b_q;/*大きい順*/
    bool aquery;
    T su = 0;
    pq(int size = inf){        aquery = size == inf;        if(!aquery){            b_q = helper_pq_size<T>(size);        }    }
    void clear() {        if(aquery) a_q.clear();        else b_q.clear();    }
    void operator+=(T v) {        if(aquery) a_q += v;        else b_q += v;    }
    T sum() {        if(aquery)return  a_q.sum();        else return b_q.sum();    }
    T top() {        if(aquery) return a_q.top();        else return b_q.top();    }
    void pop() {        if(aquery) a_q.pop();        else b_q.pop();    }
    T poll() {        if(aquery) return a_q.poll();        else return b_q.poll();    }
    ll size() {        if(aquery) return a_q.size();        else return b_q.size();    }
};
//大きいほうからsize個残る
template<class T> struct pqg {
    helper_pqg<T> a_q;/*大きい順*/
    helper_pqg_size<T> b_q;/*大きい順*/
    bool aquery;
    T su = 0;
    pqg(int size = inf) {        aquery = size == inf;        if (!aquery) {            b_q = helper_pqg_size<T>(size);        }    }
    void clear() {        if(aquery) a_q.clear();        else b_q.clear();    }
    void operator+=(T v) {if(aquery) a_q += v;else b_q += v;}
    T sum() {        if(aquery)return  a_q.sum();        else return b_q.sum();    }
    T top() {        if(aquery) return a_q.top();        else return b_q.top();    }
    void pop() {        if(aquery) a_q.pop();        else b_q.pop();    }
    T poll() {        if(aquery) return a_q.poll();        else return b_q.poll();    }
    ll size() {        if(aquery) return a_q.size();        else return b_q.size();    }
};

#define pqi pq<ll>
#define pqgi pqg<ll>
//マクロ 繰り返し
//↓@オーバーロード隔離

#define rep1(n) for(ll rep1i = 0,rep1lim=n; rep1i < rep1lim ; ++rep1i)
#define rep2(i, n) for(ll i = 0,rep2lim=n; i < rep2lim ; ++i)
#define rep3(i, m, n) for(ll i = m,rep3lim=n; i < rep3lim ; ++i)
#define rep4(i, m, n, ad) for(ll i = m,rep4lim=n; i < rep4lim ; i+= ad)
//逆順 閉区間
#define rer2(i, n) for(ll i = n; i >= 0 ; i--)
#define rer3(i, m, n) for(ll i = m,rer3lim=n; i >= rer3lim ; i--)
#define rer4(i, m, n, dec) for(ll i = m,rer4lim=n; i >= rer4lim ; i-=dec)
//ループを一つにまとめないとフォーマットで汚くなるため
#define nex_ind1(i) i++
#define nex_ind2(i, j, J) i = (j + 1 == J) ? i + 1 : i, j = (j + 1 == J ? 0 : j + 1)
#define nex_ind3(i, j, k, J, K)i = (j + 1 == J && k + 1 == K) ? i + 1 : i, j = (k + 1 == K) ? (j + 1 == J ? 0 : j + 1) : j, k = (k + 1 == K ? 0 : k + 1)
#define nex_ind4(i, j, k, l, J, K, L) i = (j + 1 == J && k + 1 == K && l + 1 == L) ? i + 1 : i, j = (k + 1 == K && l + 1 == L) ? (j + 1 == J ? 0 : j + 1) : j, k = (l + 1 == L ?(k + 1 == K ? 0 : k + 1) : k), l = l + 1 == L ? 0 : l + 1
#define nex_ind5(i, j, k, l, m, J, K, L, M) i = (j + 1 == J && k + 1 == K && l + 1 == L && m + 1 == M) ? i + 1 : i, j = (k + 1 == K && l + 1 == L && m + 1 == M) ? (j + 1 == J ? 0 : j + 1) : j, k = (l + 1 == L && m + 1 == M ?(k + 1 == K ? 0 : k + 1) : k), l =  m + 1 == M ? l+1 == L ? 0 : l+1 : l,  m = m + 1 == M ? 0 : m + 1

#define repss2(i, I) for (int i = 0; i < I; i++)
#define repss4(i, j, I, J) for (int i = (J ? 0 : I), j = 0; i < I; nex_ind2(i, j, J))
#define repss6(i, j, k, I, J, K) for (int i = (J && K ? 0 : I), j = 0, k = 0; i < I; nex_ind3(i, j, k, J, K))
#define repss8(i, j, k, l, I, J, K, L) for (int i = (J && K && L ? 0 : I), j = 0, k = 0, l = 0; i < I; nex_ind4(i, j, k, l, J, K, L))
#define repss10(i, j, k, l, m, I, J, K, L, M)for (int i = (J && K && L && M ? 0 : I), j = 0, k = 0, l = 0, m = 0; i < I; nex_ind5(i, j, k, l, m, J, K, L, M))

//i,j,k...をnまで見る
#define reps2(i, n) repss2(i, n)
#define reps3(i, j, n) repss4(i, j, n, n)
#define reps4(i, j, k, n) repss6(i, j, k, n, n, n)
#define reps5(i, j, k, l, n) repss8(i, j, k, l, n, n, n, n)

template<class T> void nex_repv2(int &i, int &j, int &I, int &J, vector<vector<T>> &s) {    while (1) {        j++;        if (j >= J) {            j = 0;            i++;            if (i < I) { J = (int) s[i].size(); }        }        if (i >= I || J) return;    }}
template<class T> void nex_repv3(int &i, int &j, int &k, int &I, int &J, int &K, vector<vector<vector<T>>> &s) {    while (1) {        k++;        if (k >= K) {            k = 0;            j++;            if (j >= J) {                j = 0;                i++;                if (i >= I)return;            }        }        J = (int) s[i].size();        K = (int) s[i][j].size();        if (J && K) return;    }}

#define repv_2(i, a) repss2(i, sz(a))
//正方形である必要はない
//直前を持つのとどっちが早いか
#define repv_3(i, j, a) for (int repvI = (int)a.size(), repvJ = (int)a[0].size(), i = 0, j = 0; i < repvI; nex_repv2(i,j,repvI,repvJ,a))
//箱状になっている事が要求される つまり[i] 次元目の要素数は一定
#define repv_4(i, j, k, a) for (int repvI = (int)a.size(), repvJ = (int)a[0].size(), repvK =(int)a[0][0].size(), i = 0, j = 0, k=0; i < repvI; nex_repv3(i,j,k,repvI,repvJ,repvK,a))
#define repv_5(i, j, k, l, a) repss8(i, j, k, l, sz(a), sz(a[0]), sz(a[0][0]), sz(a[0][0][0]))
#define repv_6(i, j, k, l, m, a) repss10(i, j, k, l, m, sz(a), sz(a[0]), sz(a[0][0]), sz(a[0][0][0]), sz(a[0][0][0][0]))

template<typename T> struct has_rbegin_rend { private:template<typename U> static auto check(U &&obj) -> decltype(std::rbegin(obj), std::rend(obj), std::true_type{});static std::false_type check(...);public:static constexpr bool value = decltype(check(std::declval<T>()))::value; };
template<typename T> constexpr bool has_rbegin_rend_v = has_rbegin_rend<T>::value;
template<typename Iterator> class Range { public:Range(Iterator &&begin, Iterator &&end) noexcept: m_begin(std::forward<Iterator>(begin)), m_end(std::forward<Iterator>(end)) {}Iterator begin() const noexcept { return m_begin; }Iterator end() const noexcept { return m_end; }private:const Iterator m_begin;const Iterator m_end; };
template<typename Iterator> static inline Range<Iterator> makeRange(Iterator &&begin, Iterator &&end) noexcept { return Range<Iterator>{std::forward<Iterator>(begin), std::forward<Iterator>(end)}; }
template<typename T> static inline decltype(auto) makeReversedRange(const std::initializer_list<T> &iniList) noexcept { return makeRange(std::rbegin(iniList), std::rend(iniList)); }
template<typename T, typename std::enable_if_t<has_rbegin_rend_v<T>, std::nullptr_t> = nullptr> static inline decltype(auto) makeReversedRange(T &&c) noexcept { return makeRange(std::rbegin(c), std::rend(c)); }/* rbegin(), rend()を持たないものはこっちに分岐させて，エラーメッセージを少なくする*/template<typename T, typename std::enable_if<!has_rbegin_rend<T>::value, std::nullptr_t>::type = nullptr> static inline void makeReversedRange(T &&) noexcept { static_assert(has_rbegin_rend<T>::value, "Specified argument doesn't have reverse iterator."); }

#define use_for

#define form1(st) for (auto &&form_it = st.begin(); form_it != st.end(); ++form_it)
#define form3(k, v, st) for (auto &&form_it = st.begin(); form_it != st.end(); ++form_it)
#define form4(k, v, st, r) for (auto &&form_it = st.begin(); form_it != st.end() && (*form_it).fi < r; ++form_it)
#define form5(k, v, st, l, r) for (auto &&form_it = st.lower_bound(l); form_it != st.end() && (*form_it).fi < r; ++form_it)

#define forrm1(st) for (auto &&forrm_it = st.rbegin(); forrm_it != st.rend(); ++forrm_it)
#define forrm3(k, v, st) for (auto &&forrm_it = st.rbegin(); forrm_it != st.rend(); ++forrm_it)

//向こう側で
// ++itか it = st.erase(it)とする
#define fors1(st) for (auto &&it = st.begin(); it != st.end(); )
#define fors2(v, st) for (auto &&it = st.begin(); it != st.end(); )
#define fors3(v, st, r) for (auto &&it = st.begin(); it != st.end() && (*it) < r; )
#define fors4(v, st, l, r) for (auto &&it = st.lower_bound(l); it != st.end() && (*it) < r; )

#ifdef use_for
#define forslr3(st, a, b) for (auto &&forslr_it = st.begin(); forslr_it != st.end(); ++forslr_it)
#define forslr4(v, st, a, b) for (auto &&forslr_it = st.begin(); forslr_it != st.end(); ++forslr_it)
#define forslr5(v, st, r, a, b) for (auto &&forslr_it = st.begin(); forslr_it != st.end() && (*forslr_it) < r; ++forslr_it)
#define forslr6(v, st, l, r, a, b) for (auto &&forslr_it = st.lower_bound(l); forslr_it != st.end() && (*forslr_it) < r; ++forslr_it)
#endif

#define fora_f_init_2(a, A) ;
#define fora_f_init_3(fora_f_i, a, A) auto &&a = A[fora_f_i];
#define fora_f_init_4(a, b, A, B) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i];
#define fora_f_init_5(fora_f_i, a, b, A, B) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i];
#define fora_f_init_6(a, b, c, A, B, C) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i]; auto &&c = C[fora_f_i];
#define fora_f_init_7(fora_f_i, a, b, c, A, B, C) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i]; auto &&c = C[fora_f_i];
#define fora_f_init_8(a, b, c, d, A, B, C, D) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i]; auto &&c = C[fora_f_i]; auto && d = D[fora_f_i];
#define fora_f_init_9(fora_f_i, a, b, c, d, A, B, C, D) auto &&a = A[fora_f_i]; auto &&b = B[fora_f_i]; auto &&c = C[fora_f_i]; auto && d = D[fora_f_i];
#define fora_f_init(...) over9(__VA_ARGS__,fora_f_init_9, fora_f_init_8, fora_f_init_7, fora_f_init_6, fora_f_init_5, fora_f_init_4, fora_f_init_3, fora_f_init_2)(__VA_ARGS__)

#ifdef use_for
#define forr_init_2(a, A) auto &&a = A[forr_i];
#define forr_init_3(forr_i, a, A) auto &&a = A[forr_i];
#define forr_init_4(a, b, A, B) auto &&a = A[forr_i]; auto &&b = B[forr_i];
#define forr_init_5(forr_i, a, b, A, B) auto &&a = A[forr_i]; auto &&b = B[forr_i];
#define forr_init_6(a, b, c, A, B, C) auto &&a = A[forr_i]; auto &&b = B[forr_i]; auto &&c = C[forr_i];
#define forr_init_7(forr_i, a, b, c, A, B, C) auto &&a = A[forr_i]; auto &&b = B[forr_i]; auto &&c = C[forr_i];
#define forr_init_8(a, b, c, d, A, B, C, D) auto &&a = A[forr_i]; auto &&b = B[forr_i]; auto &&c = C[forr_i]; auto && d = D[forr_i];
#define forr_init_9(forr_i, a, b, c, d, A, B, C, D) auto &&a = A[forr_i]; auto &&b = B[forr_i]; auto &&c = C[forr_i]; auto && d = D[forr_i];
#define forr_init(...) over9(__VA_ARGS__, forr_init_9, forr_init_8, forr_init_7, forr_init_6, forr_init_5, forr_init_4, forr_init_3, forr_init_2)(__VA_ARGS__)

#define forp_init3(k, v, S) auto &&k = S[forp_i].first;auto &&v = S[forp_i].second;
#define forp_init4(forp_i, k, v, S) auto &&k = S[forp_i].first;auto &&v = S[forp_i].second;
#define forp_init(...) over4(__VA_ARGS__,forp_init4,forp_init3,forp_init2,forp_init1)(__VA_ARGS__)

#define form_init(k, v, ...) auto &&k = (*form_it).fi;auto &&v = (*form_it).se;
#define forrm_init(k, v, ...) auto &&k = (*forrm_it).fi;auto &&v = (*forrm_it).se;
#endif
#define fors_init(v, ...) auto &&v = (*it);

#ifdef use_for
#define forlr_init(a, A, ngl, ngr) auto a = A[forlr_i]; auto prev = forlr_i ? A[forlr_i-1] : ngl;auto next = forlr_i+1< rep2lim? A[forlr_i+1] : ngr;
#define forslr_init4(a, A, ngl, ngr) auto a = (*forslr_it); auto prev = (forslr_it!=A.begin())? (*std::prev(forslr_it)) : ngl;auto next = (forslr_it!=std::prev(A.end()))? (*std::next(forslr_it)) : ngr;
#define forslr_init5(a, A, r, ngl, ngr) auto a = (*forslr_it); auto prev = (forslr_it!=A.begin())? (*std::prev(forslr_it)) : ngl;auto next = (forslr_it!=std::prev(A.end()))? (*std::next(forslr_it)) : ngr;
#define forslr_init6(a, A, l, r, ngl, ngr) auto a = (*forslr_it); auto prev = (forslr_it!=A.begin())? (*std::prev(forslr_it)) : ngl;auto next = (forslr_it!=std::prev(A.end()))? (*std::next(forslr_it)) : ngr;
#define forslr_init(...)  over6(__VA_ARGS__,forslr_init6,forslr_init5,forslr_init4)(__VA_ARGS__);
#endif

//こうしないとmapがおかしくなる
#define fora_f_2(a, A) for(auto&& a : A)
#define fora_f_3(fora_f_i, a, A) rep(fora_f_i, sz(A))
#define fora_f_4(a, b, A, B) rep(fora_f_i, sz(A))
#define fora_f_5(fora_f_i, a, b, A, B) rep(fora_f_i, sz(A))
#define fora_f_6(a, b, c, A, B, C) rep(fora_f_i, sz(A))
#define fora_f_7(fora_f_i, a, b, c, A, B, C) rep(fora_f_i, sz(A))
#define fora_f_8(a, b, c, d, A, B, C, D) rep(fora_f_i, sz(A))
#define fora_f_9(fora_f_i, a, b, c, d, A, B, C, D) rep(fora_f_i, sz(A))

#ifdef use_for
#define forr_2(a, A) rer(forr_i, sz(A)-1)
#define forr_3(forr_i, a, A) rer(forr_i, sz(A)-1)
#define forr_4(a, b, A, B) rer(forr_i, sz(A)-1)
#define forr_5(forr_i, a, b, A, B) rer(forr_i, sz(A)-1)
#define forr_6(a, b, c, A, B, C) rer(forr_i, sz(A)-1)
#define forr_7(forr_i, a, b, c, A, B, C) rer(forr_i, sz(A)-1)
#define forr_8(a, b, c, d, A, B, C, D) rer(forr_i, sz(A)-1)
#define forr_9(forr_i, a, b, c, d, A, B, C, D) rer(forr_i, sz(A)-1)
#endif
//↑@オーバーロード隔離

//rep系はインデックス、for系は中身
#define rep(...) over4(__VA_ARGS__,rep4,rep3,rep2,rep1)(__VA_ARGS__)
#define rer(...) over4(__VA_ARGS__,rer4,rer3,rer2,)(__VA_ARGS__)

//自分込みで残りがREM以上の間ループを回す
#define rem(i, N, REM) for (int i = 0; i < N - REM + 1; i++)

//char用のrep
#define repc(i, m, n) for(char i = m,repc3lim=n; i < repc3lim ; ++i)
//i,j,k...をnまで見る
#define reps(...) over5(__VA_ARGS__,reps5,reps4,reps3,reps2,)(__VA_ARGS__)
#define repss(...) over10(__VA_ARGS__, repss10, a, repss8, a, repss6, a, repss4, a, repss2) (__VA_ARGS__)
//vectorのindexを走査する
//repv(i,j,vvi)
#define repv(...) over6(__VA_ARGS__,repv_6,repv_5,repv_4,repv_3,repv_2,)(__VA_ARGS__)
#define rerv(i, A) for (int i = sz(A)-1; i >= 0 ; i--)
//repvn(dp) nは次元
#define repv1(a) repv(i, a)
#define repv2(a) repv(i, j, a)
#define repv3(a) repv(i, j, k, a)
#define repv4(a) repv(i, j, k, l, a)

#define fora_f(...) over9(__VA_ARGS__, fora_f_9, fora_f_8, fora_f_7, fora_f_6, fora_f_5, fora_f_4, fora_f_3, fora_f_2)(__VA_ARGS__)
#define forr(...) over9(__VA_ARGS__, forr_9, forr_8, forr_7, forr_6, forr_5, forr_4, forr_3, forr_2)(__VA_ARGS__)
//0~N-2まで見る

#define forar_init(v, rv, A) auto &&v = A[forar_i]; auto && rv = A[forar_i+1];
#define forar(v, rv, A) rep(forar_i, sz(A) - 1)

///c++17
//template<size_t M_SZ, bool indexed, class Iterator, class T, class U=int, class V=int, class W=int>
//class ite_vec_merge : public Iterator {    std::size_t i = 0;    vector<T> &A;vector<U> &B;vector<V> &C;vector<W> &D;public :    ite_vec_merge(Iterator ita, vector<T> &A) : Iterator(ita), A(A), B(A), C(A), D(A) {}    ite_vec_merge(Iterator ita, vector<T> &A, vector<U> &B) : Iterator(ita), A(A), B(B), C(A), D(A) {}    ite_vec_merge(Iterator ita, vector<T> &A, vector<U> &B, vector<V> &C) : Iterator(ita), A(A), B(B), C(C), D(A) {}    ite_vec_merge(Iterator ita, vector<T> &A, vector<U> &B, vector<V> &C, vector<W> &D) : Iterator(ita), A(A), B(B), C(C), D(D) {}    auto &operator++() {        ++i;        this->Iterator::operator++();        return *this;    }    auto operator*() const noexcept {        if constexpr(!indexed && M_SZ == 1) {            return tuple<T &>(A[i]);        } else if constexpr(!indexed && M_SZ == 2) {            return tuple<T &, U &>(A[i], B[i]);        } else if constexpr(!indexed && M_SZ == 3) {            return tuple<T &, U &, V &>(A[i], B[i], C[i]);        } else if constexpr(!indexed && M_SZ == 4) {            return tuple<T &, U &, V &, W &>(A[i], B[i], C[i], D[i]);        } else if constexpr(indexed && M_SZ == 1) {            return tuple<int, T &>(i, A[i]);        } else if constexpr(indexed && M_SZ == 2) {            return tuple<int, T &, U &>(i, A[i], B[i]);        } else if constexpr(indexed && M_SZ == 3) {            return tuple<int, T &, U &, V &>(i, A[i], B[i], C[i]);        } else if constexpr(indexed && M_SZ == 4) {            return tuple<int, T &, U &, V &, W &>(i, A[i], B[i], C[i], D[i]);        } else {            assert(0);            return tuple<int>(i);        }    }};
//template<size_t M_SZ, bool indexed, class T, class U=int, class V=int, class W=int>
//class vec_merge {    vector<T> &a;vector<U> &b;vector<V> &c;vector<W> &d;public :    vec_merge(vector<T> &a) : a(a), b(a), c(a), d(a) {}    vec_merge(vector<T> &a, vector<U> &b) : a(a), b(b), c(a), d(a) {}    vec_merge(vector<T> &a, vector<U> &b, vector<V> &c) : a(a), b(b), c(c), d(a) {}    vec_merge(vector<T> &a, vector<U> &b, vector<V> &c, vector<W> &d) : a(a), b(b), c(c), d(d) {}    auto begin() const {        if constexpr(M_SZ == 1) {            return ite_vec_merge<M_SZ, indexed, decltype(std::begin(a)), T, U, V, W>{std::begin(a), a};        } else if constexpr(M_SZ == 2) {            return ite_vec_merge<M_SZ, indexed, decltype(std::begin(a)), T, U, V, W>{std::begin(a), a, b};        } else if constexpr(M_SZ == 3) {            return ite_vec_merge<M_SZ, indexed, decltype(std::begin(a)), T, U, V, W>{std::begin(a), a, b, c};        } else if constexpr(M_SZ == 4) {            return ite_vec_merge<M_SZ, indexed, decltype(std::begin(a)), T, U, V, W>{std::begin(a), a, b, c, d};        } else {            assert(0);            return ite_vec_merge<M_SZ, indexed, decltype(std::begin(a)), T, U, V, W>{std::begin(a), a};        }    }    auto end() const {        if constexpr(M_SZ == 1) {            return ite_vec_merge<M_SZ, indexed, decltype(std::end(a)), T, U, V, W>{std::end(a), a};        } else if constexpr(M_SZ == 2) {            return ite_vec_merge<M_SZ, indexed, decltype(std::end(a)), T, U, V, W>{std::end(a), a, b};        } else if constexpr(M_SZ == 3) {            return ite_vec_merge<M_SZ, indexed, decltype(std::end(a)), T, U, V, W>{std::end(a), a, b, c};        } else if constexpr(M_SZ == 4) {            return ite_vec_merge<M_SZ, indexed, decltype(std::end(a)), T, U, V, W>{std::end(a), a, b, c, d};        } else {            assert(0);            return ite_vec_merge<M_SZ, indexed, decltype(std::end(a)), T, U, V, W>{std::end(a), a};        }    }};
#define fora_2(a, A) for(auto&& a : A)
//#define fora_3(i, a, A) for(auto[i, a] : vec_merge<1, true, decl_t(A)>(A))
//#define fora_4(a, b, A, B) for(auto[a, b] : vec_merge<2, false, decl_t(A), decl_t(B)>(A, B))
//#define fora_5(i, a, b, A, B)  for(auto[i, a, b] : vec_merge<2, true, decl_t(A), decl_t(B)>(A, B))
//#define fora_6(a, b, c, A, B, C)  for(auto[a, b, c] : vec_merge<3, false, decl_t(A), decl_t(B), decl_t(C)>(A, B, C))
//#define fora_7(i, a, b, c, A, B, C)  for(auto[i, a, b, c] : vec_merge<3, true, decl_t(A), decl_t(B), decl_t(C)>(A, B, C))
//#define fora_8(a, b, c, d, A, B, C, D) for(auto[a, b, c, d] : vec_merge<4, false, decl_t(A), decl_t(B), decl_t(C), decl_t(D)>(A, B, C, D))
//#define fora_9(i, a, b, c, d, A, B, C, D) for(auto[i, a, b, c, d] : vec_merge<4, true, decl_t(A), decl_t(B), decl_t(C), decl_t(D)>(A, B, C, D))
//構造化束縛ver
//1e5要素で40ms程度
//遅いときはfora_fを使う
//todo c++17
#define fora(...) over9(__VA_ARGS__, fora_9, fora_8, fora_7, fora_6, fora_5, fora_4, fora_3, fora_2)(__VA_ARGS__)
///c++17
///
//#define forr(v, a) for(auto&& v : makeReversedRange(a))
//参照を取らない
#ifdef use_for
template<class U> vector<U> to1d(vector<U> &a) { return a; }template<class U> auto to1d(vector<vector<U>> &a) {    vector<U> res;    for (auto &&a1 : a)for (auto &&a2 : a1)res.push_back(a2);    return res;}template<class U> vector<U> to1d(vector<vector<vector<U>>> &a) {    vector<U> res;    for (auto &&a1 : a)for (auto &&a2 : a1)for (auto &&a3 : a2) res.push_back(a3);    return res;}template<class U> vector<U> to1d(vector<vector<vector<vector<U>>>> &a) {    vector<U> res;    for (auto &&a1 : a)for (auto &&a2 : a1)for (auto &&a3 : a2) for (auto &&a4 : a3)res.push_back(a4);    return res;}template<class U> vector<U> to1d(vector<vector<vector<vector<vector<U>>>>> &a) {    vector<U> res;    for (auto &&a1 : a)for (auto &&a2 : a1)for (auto &&a3 : a2) for (auto &&a4 : a3)for (auto &&a5 : a4)res.push_back(a5);    return res;}template<class U> vector<U> to1d(vector<vector<vector<vector<vector<vector<U>>>>>> &a) {    vector<U> res;    for (auto &&a1 : a)for (auto &&a2 : a1)for (auto &&a3 : a2) for (auto &&a4 : a3)for (auto &&a5 : a4)for (auto &&a6 : a5)res.push_back(a6);    return res;}
#define forv(a, b) for(auto a : to1d(b))
//インデックスを前後含めて走査
#define ring(i, s, len) for (int i = s, prev = (s == 0) ? len - 1 : s - 1, next = (s == len - 1) ? 0 : s + 1, cou = 0; cou < len; cou++, prev = i, i = next, next = (next == len - 1) ? 0 : next + 1)
//値と前後を見る
#define ringv(v, d) index_=0;for (auto prev = d[sz(d)-1],next= (int)d.size()>1?d[1]:d[0],v = d[0]; index_ < sz(d); index_++, prev = v, v = next, next = (index_>=sz(d)-1?d[0]:d[index_+1]))
// 左右をnext prevで見る 0の左と nの右
#define forlr(v, d, banpei_l, banpei_r) rep(forlr_i,sz(d))
#endif

#define form(...) over5(__VA_ARGS__,form5,form4,form3,form2,form1)(__VA_ARGS__)
#define forrm(...) over5(__VA_ARGS__,forrm5,forrm4,forrm3,forrm2,forrm1)(__VA_ARGS__)
#define fors(...) over4(__VA_ARGS__,fors4,fors3,fors2,fors1)(__VA_ARGS__)
#define forslr(...) over6(__VA_ARGS__,forslr6,forslr5,forslr4,forslr3)(__VA_ARGS__)
#define forp3(k, v, st)  rep(forp_i,sz(st))
#define forp4(forp_i, k, v, st) rep(forp_i,sz(st))
#define forp(...) over4(__VA_ARGS__,forp4,forp3)(__VA_ARGS__)


//マクロ 定数
#define k3 1010
#define k4 10101
#define k5 101010
#define k6 1010101
#define k7 10101010

const double PI = 3.1415926535897932384626433832795029L;


constexpr bool ev(ll a) { return !(a & 1); }
constexpr bool od(ll a) { return (a & 1); }
//@拡張系 こう出来るべきというもの

//埋め込み 存在を意識せずに機能を増やされているもの
namespace std {
    template<> class hash<std::pair<signed, signed>> { public:size_t operator()(const std::pair<signed, signed> &x) const { return hash<ll>()(((ll) x.first << 32) | x.second); }};
    template<> class hash<std::pair<ll, ll>> { public:/*大きいllが渡されると、<<32でオーバーフローするがとりあえず問題ないと判断*/size_t operator()(const std::pair<ll, ll> &x) const { return hash<ll>()(((ll) x.first << 32) | x.second); }};
}
//stream まとめ
istream &operator>>(istream &iss, P &a) {    iss >> a.first >> a.second;    return iss;}template<typename T> istream &operator>>(istream &iss, vector<T> &vec_) {    for (T &x: vec_) iss >> x;    return iss;}template<class T, class U> ostream &operator<<(ostream &os, pair<T, U> p) {    os << p.fi << " " << p.se;    return os;}ostream &operator<<(ostream &os, T p) {    os << p.f << " " << p.s << " " << p.t;    return os;}ostream &operator<<(ostream &os, F p) {    os << p.a << " " << p.b << " " << p.c << " " << p.d;    return os;}template<typename T> ostream &operator<<(ostream &os, vector<T> &vec_) {    for (ll i = 0; i < vec_.size(); ++i)os << vec_[i] << (i + 1 == vec_.size() ? "" : " ");    return os;}template<typename T> ostream &operator<<(ostream &os, vector<vector<T>> &vec_) {    for (ll i = 0; i < vec_.size(); ++i) {        for (ll j = 0; j < vec_[i].size(); ++j) { os << vec_[i][j] << " "; }        os << endl;    }    return os;}template<typename T, typename U> ostream &operator<<(ostream &os, map<T, U> &m) {    os << endl;    for (auto &&v:m) os << v << endl;    return os;}template<class T> ostream &operator<<(ostream &os, set<T> s) {    fora_f(v, s){fora_f_init(v, s); { os << v << " "; }}    return os;}template<class T> ostream &operator<<(ostream &os, mset<T> s) {    fora_f(v, s) {fora_f_init(v, s); os << v << " "; }    return os;}template<class T> ostream &operator<<(ostream &os, deque<T> a) {    fora_f(v, a){fora_f_init(v, a);os << v << " "; }   return os;}ostream &operator<<(ostream &os, vector<vector<char>> &vec_) {    rep(h, sz(vec_)) {        rep(w, sz(vec_[0])) { os << vec_[h][w]; }        os << endl;    }    return os;}
//template<class T,class U>ostream &operator<<(ostream &os, vector<pair<T,U>>& a) {fora_f(v,a)os<<v<<endl;return os;}
template<typename W, typename H> void resize(W &vec_, const H head) { vec_.resize(head); }
template<typename W, typename H, typename ... T> void resize(W &vec_, const H &head, const T ... tail) {vec_.resize(head);for (auto &v: vec_)resize(v, tail...);}
#define use_for_each  //_each _all_of _any_of _none_of _find_if _rfind_if _contains _count_if _erase_if _entry_if
#ifdef use_for_each
template<typename T, typename F> bool all_of2(const T &v, F f) { return f(v); }
template<typename T, typename F> bool all_of2(const vector<T> &v, F f) {rep(i, sz(v)) { if (!all_of2(v[i], f))return false; }return true;}
template<typename T, typename F> bool any_of2(const T &v, F f) { return f(v); }
template<typename T, typename F> bool any_of2(const vector<T> &v, F f) {    rep(i, sz(v)) { if (any_of2(v[i], f))return true; }    return false;}
template<typename T, typename F> bool none_of2(const T &v, F f) { return f(v); }
template<typename T, typename F> bool none_of2(const vector<T> &v, F f) {    rep(i, sz(v)) { if (none_of2(v[i], f))return false; }    return true;}
template<typename T, typename F> bool find_if2(const T &v, F f) { return f(v); }
template<typename T, typename F> ll find_if2(const vector<T> &v, F f) {    rep(i, sz(v)) { if (find_if2(v[i], f))return i; }    return sz(v);}
template<typename T, typename F> bool rfind_if2(const T &v, F f) { return f(v); }
template<typename T, typename F> ll rfind_if2(const vector<T> &v, F f) {    rer(i, sz(v) - 1) { if (rfind_if2(v[i], f))return i; }    return -1;}
template<class T> bool contains(const string &s, const T &v) { return s.find(v) != string::npos; }
template<typename T> bool contains(const vector<T> &v, const T &val) { return std::find(v.begin(), v.end(), val) != v.end(); }
template<typename T, typename F> bool contains_if2(const vector<T> &v, F f) { return find_if(v.begin(), v.end(), f) != v.end(); }
template<typename T, typename F> ll count_if2(const T &v, F f) { return f(v); }
template<typename T, typename F> ll count_if2(const vector<T> &vec_, F f) {    ll ret = 0;    fora_f(v, vec_){fora_f_init(v, vec_);ret += count_if2(v, f);}    return ret;}
//template<typename T, typename F> void for_each2(T &v, F f) { f(v); }
//template<typename T, typename F> void for_each2(vector<T> &vec_, F f) { fora_f(v, vec_)for_each2(v, f); }
template<typename T, typename F> void for_each2(vector<T> &a, F f) { rep(i, sz(a))f(a[i]); }
template<typename T, typename F> void for_each2(vector<vector<T> > &a, F f) { rep(i, sz(a))rep(j, sz(a[i]))f(a[i][j]); }
template<typename T, typename F> void for_each2(vector<vector<vector<T> > > &a, F f) { rep(i, sz(a))rep(j, sz(a[i]))rep(k, sz(a[i][j]))f(a[i][j][k]); }

template<typename W> ll count_od(const vector<W> &a) { return count_if2(a, [](ll v) { return v & 1; }); }
template<typename W> ll count_ev(const vector<W> &a) { return count_if2(a, [](ll v) { return !(v & 1); }); }
//削除した後のvectorを返す
template<typename T, typename F> vector<T> erase_if2(const vector<T> &v, F f) {    vector<T> nv;    rep(i, sz(v)) { if (!f(v[i])) { nv.push_back(v[i]); }}    return nv;}
template<typename T, typename F> vector<vector<T>> erase_if2(const vector<vector<T>> &v, F f) {    vector<vector<T>> res;    rep(i, sz(v)) { res[i] = erase_if2(v[i], f); }    return res;}
template<typename T, typename F> vector<T> l_erase_if2(const vector<T> &v, F f) {    vector<T> nv;    rep(i, sz(v)) { if (!f(v[i])) { nv.push_back(v[i]); }}    return nv;}

template<typename T, typename F> vector<T> entry_if2(const vector<T> &v, F f) {    vector<T> nv;    rep(i, sz(v)) { if (f(v[i])) { nv.push_back(v[i]); }}    return nv;}
template<typename T, typename F> vector<vector<T>> entry_if2(const vector<vector<T>> &v, F f) {    vector<vector<T>> res;    rep(i, sz(v)) { res[i] = entry_if2(v[i], f); }    return res;}
template<typename T, typename F> vector<T> l_entry_if2(const vector<T> &v, F f) {    vector<T> nv;    rep(i, sz(v)) { if (f(v[i])) { nv.push_back(v[i]); }}    return nv;}

template<typename T, typename F> ll l_rfind_if(const vector<T> &v, F f) {    rer(i, sz(v) - 1) { if (f(v[i]))return i; }    return -1;}
template<typename T, typename F> bool l_contains_if(const vector<T> &v, F f) {    rer(i, sz(v) - 1) { if (f(v[i]))return true; }    return false;}
template<class A, class B, class C> auto t_all_of(A a, B b, C c) { return std::all_of(a, b, c); }
template<class A, class B, class C> auto t_any_of(A a, B b, C c) { return std::any_of(a, b, c); }
template<class A, class B, class C> auto t_none_of(A a, B b, C c) { return std::none_of(a, b, c); }
template<class A, class B, class C> auto t_find_if(A a, B b, C c) { return std::find_if(a, b, c); }
template<class A, class B, class C> auto t_count_if(A a, B b, C c) { return std::count_if(a, b, c); }

#define all_of_s__2(a, right) (t_all_of(all(a),lamr(right)))
#define all_of_s__3(a, v, siki) (t_all_of(all(a),[&](auto v){return siki;}))
#define all_of_s(...) over3(__VA_ARGS__,all_of_s__3,all_of_s__2)(__VA_ARGS__)

//all_of(A, %2);
//all_of(A, a, a%2);
#define all_of__2(a, right) all_of2(a,lamr(right))
#define all_of__3(a, v, siki) all_of2(a,[&](auto v){return siki;})
#define all_of(...) over3(__VA_ARGS__,all_of__3,all_of__2)(__VA_ARGS__)
#define all_of_f(a, f) all_of2(a,f)

#define any_of_s__2(a, right) (t_any_of(all(a),lamr(right)))
#define any_of_s__3(a, v, siki) (t_any_of(all(a),[&](auto v){return siki;}))
#define any_of_s(...) over3(__VA_ARGS__,any_of_s__3,any_of_s__2)(__VA_ARGS__)

#define any_of__2(a, right) any_of2(a,lamr(right))
#define any_of__3(a, v, siki) any_of2(a,[&](auto v){return siki;})
#define any_of(...) over3(__VA_ARGS__,any_of__3,any_of__2)(__VA_ARGS__)
#define any_of_f(a, f) any_of2(a,f)

#define none_of_s__2(a, right) (t_none_of(all(a),lamr(right)))
#define none_of_s__3(a, v, siki) (t_none_of(all(a),[&](auto v){return siki;}))
#define none_of_s(...) over3(__VA_ARGS__,none_of_s__3,none_of_s__2)(__VA_ARGS__)

#define none_of__2(a, right) none_of2(a,lamr(right))
#define none_of__3(a, v, siki) none_of2(a,[&](auto v){return siki;})
#define none_of(...) over3(__VA_ARGS__,none_of__3,none_of__2)(__VA_ARGS__)
#define none_of_f(a, f) none_of2(a,f)

#define find_if_s__2(a, right) (t_find_if(all(a),lamr(right))-a.begin())
#define find_if_s__3(a, v, siki) (t_find_if(all(a),[&](auto v){return siki;})-a.begin())
#define find_if_s(...) over3(__VA_ARGS__,find_if_s__3,find_if_s__2)(__VA_ARGS__)

#define find_if__2(a, right) find_if2(a,lamr(right))
#define find_if__3(a, v, siki) find_if2(a,[&](auto v){return siki;})
#define find_if(...) over3(__VA_ARGS__,find_if__3,find_if__2)(__VA_ARGS__)
#define find_if_f(a, f) find_if2(a,f)

#define rfind_if_s__2(a, right) l_rfind_if(a, lamr(right))
#define rfind_if_s__3(a, v, siki) l_rfind_if(a, [&](auto v){return siki;})
#define rfind_if_s(...) over3(__VA_ARGS__,rfind_if_s__3,rfind_if_s__2)(__VA_ARGS__)

#define rfind_if__2(a, right) rfind_if2(a,lamr(right))
#define rfind_if__3(a, v, siki) rfind_if2(a,[&](auto v){return siki;})
#define rfind_if(...) over3(__VA_ARGS__,rfind_if__3,rfind_if__2)(__VA_ARGS__)
#define rfind_if_f(a, f) rfind_if2(a,f)

#define contains_if_s__2(a, right) l_contains_if(a, lamr(right))
#define contains_if_s__3(a, v, siki) l_contains_if(a, [&](auto v){return siki;})
#define contains_if_s(...) over3(__VA_ARGS__,contains_if_s__3,contains_if_s__2)(__VA_ARGS__)

#define contains_if__2(a, right) contains_if2(a,lamr(right))
#define contains_if__3(a, v, siki) contains_if2(a,[&](auto v){return siki;})
#define contains_if(...) over3(__VA_ARGS__,contains_if__3,contains_if__2)(__VA_ARGS__)
#define contains_if_f(a, f) contains_if2(a,f)

#define count_if_s__2(a, right) (t_count_if(all(a),lamr(right)))
#define count_if_s__3(a, v, siki) (t_count_if(all(a),[&](auto v){return siki;}))
#define count_if_s(...) over3(__VA_ARGS__,count_if_s__3,count_if_s__2)(__VA_ARGS__)

#define count_if__2(a, right) count_if2(a,lamr(right))
#define count_if__3(a, v, siki) count_if2(a,[&](auto v){return siki;})
#define count_if(...) over3(__VA_ARGS__,count_if__3,count_if__2)(__VA_ARGS__)
#define count_if_f(a, f) count_if2(a,f)

//vector<vi>で、viに対して操作
#define for_each_s__2(a, right) do{fora_f(v,a){fora_f_init(v, a); v right;}}while(0)
#define for_each_s__3(a, v, shori) do{fora_f(v,a){fora_f_init(v, a); shori;}}while(0)
#define for_each_s(...) over3(__VA_ARGS__,for_each_s__3,for_each_s__2)(__VA_ARGS__)

//vector<vi>で、intに対して操作
#define for_each__2(a, right) for_each2(a,lamr(right))
#define for_each__3(a, v, shori) for_each2(a,[&](auto& v){shori;})
#define for_each(...) over3(__VA_ARGS__,for_each__3,for_each__2)(__VA_ARGS__)
#define for_each_f(a, f) for_each2(a, f);

#define for_eached__2(a, right) [&](auto a) {auto b = a;for_each(b, right);return b; }(a)
#define for_eached__3(a, v, shori) [&](auto a) {auto b = a;for_each(b, v, shori);return b; }(a)
#define for_eached(...) over3(__VA_ARGS__,for_eached__3,for_eached__2)(__VA_ARGS__)
#define for_eached_f(a, f) for_eached2(a, f);


#define erase_if_s__2(a, right) l_erase_if2(a,lamr(right))
#define erase_if_s__3(a, v, siki) l_erase_if2(a,[&](auto v){return siki;})
#define erase_if_s(...) over3(__VA_ARGS__,erase_if_s__3,erase_if_s__2)(__VA_ARGS__)


#define erase_if__2(a, right) erase_if2(a,lamr(right))
#define erase_if__3(a, v, siki) erase_if2(a,[&](auto v){return siki;})
#define erase_if(...) over3(__VA_ARGS__,erase_if__3,erase_if__2)(__VA_ARGS__)
#define erase_if_f(a, f) erase_if2(a,f)

#define entry_if_s__2(a, right) l_entry_if2(a,lamr(right))
#define entry_if_s__3(a, v, siki) l_entry_if2(a,[&](auto v){return siki;})
#define entry_if_s(...) over3(__VA_ARGS__,entry_if_s__3,entry_if_s__2)(__VA_ARGS__)


#define entry_if__2(a, right) entry_if2(a,lamr(right))
#define entry_if__3(a, v, siki) entry_if2(a,[&](auto v){return siki;})
#define entry_if(...) over3(__VA_ARGS__,entry_if__3,entry_if__2)(__VA_ARGS__)
#define entry_if_f(a, f) entry_if2(a,f)

#endif

template<class T, class U, class W> void replace(vector<W> &a, T key, U v) { rep(i, sz(a))if (a[i] == key)a[i] = v; }
template<class T, class U, class W> void replace(vector<vector<W>> &A, T key, U v) { rep(i, sz(A))replace(A[i], key, v); }
void replace(str &a, char key, str v) { if (v == "")a.erase(remove(all(a), key), a.end()); }
void replace(str &a, char key, char v) { replace(all(a), key, v); }
//keyと同じかどうか01で置き換える
template<class T, class U> void replace(vector<T> &a, U k) { rep(i, sz(a)) a[i] = a[i] == k; }
template<class T, class U> void replace(vector<vector<T >> &a, U k) { rep(i, sz(a))rep(j, sz(a[0])) a[i][j] = a[i][j] == k; }
//template<class T> void replace(T &a) { replace(a, '#'); }
void replace(str &a) {    int dec = 0;    if ('a' <= a[0] && a[0] <= 'z')dec = 'a';    if ('A' <= a[0] && a[0] <= 'Z')dec = 'A';    fora_f(v, a){fora_f_init(v, a); v -= dec;}}
void replace(str &a, str key, str v) {    stringstream t;    ll kn = sz(key);    std::string::size_type Pos(a.find(key));    ll l = 0;    while (Pos != std::string::npos) {        t << a.substr(l, Pos - l);        t << v;        l = Pos + kn;        Pos = a.find(key, Pos + kn);    }    t << a.substr(l, sz(a) - l);    a = t.str();}

template<class T> bool includes(vector<T> &a, vector<T> &b) {    vi c = a;    vi d = b;    sort(all(c));    sort(all(d));    return includes(all(c), all(d));}
template<class T> bool is_permutation(vector<T> &a, vector<T> &b) { return is_permutation(all(a), all(b)); }
template<class T> bool next_permutation(vector<T> &a) { return next_permutation(all(a)); }
void iota(vector<ll> &ve, ll s, ll n) {    ve.resize(n);    iota(all(ve), s);}
vi iota(ll s, ll len) {    vi ve(len);    iota(all(ve), s);    return ve;}
template<class A, class B> auto vtop(vector<A> &a, vector<B> &b) {    assert(sz(a) == sz(b));    /*stringを0で初期化できない  */  vector<pair<A, B>> res;    rep(i, sz(a))res.eb(a[i], b[i]);    return res;}
template<class A, class B> void ptov(vector<pair<A, B>> &p, vector<A> &a, vector<B> &b) {    a.resize(sz(p)), b.resize(sz(p));    rep(i, sz(p))a[i] = p[i].fi, b[i] = p[i].se;}
template<class A, class B, class C> auto vtot(vector<A> &a, vector<B> &b, vector<C> &c) {    assert(sz(a) == sz(b) && sz(b) == sz(c));    vector<T2<A, B, C>> res;    rep(i, sz(a))res.eb(a[i], b[i], c[i]);    return res;}
template<class A, class B, class C, class D> auto vtof(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    assert(sz(a) == sz(b) && sz(b) == sz(c) && sz(c) == sz(d));    vector<F2<A, B, C, D>> res;    rep(i, sz(a))res.eb(a[i], b[i], c[i], d[i]);    return res;}

template<class T> void sort(vector<T> &a) { sort(all(a)); }
template<class T> void rsort(vector<T> &a) { sort(all(a), greater<T>()); };
template<class A, class B> void sortp(vector<A> &a, vector<B> &b) {    auto c = vtop(a, b);    sort(c);    rep(i, sz(a)) a[i] = c[i].fi, b[i] = c[i].se;}
template<class A, class B> void rsortp(vector<A> &a, vector<B> &b) {    auto c = vtop(a, b);    rsort(c);    rep(i, sz(a))a[i] = c[i].first, b[i] = c[i].second;}
template<class A, class B, class C> void sortt(vector<A> &a, vector<B> &b, vector<C> &c) {    auto d = vtot(a, b, c);    sort(d);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C> void rsortt(vector<A> &a, vector<B> &b, vector<C> &c) {    auto d = vtot(a, b, c);    rsort(d);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class... T, class U> auto sorted(U head, T... a) {    sort(head, a...);    return head;}
template<class... T, class U> auto rsorted(U head, T... a) {    rsort(head, a...);    return head;}
//sortindex 元のvectorはソートしない
template<class T> vi sorti(vector<T> &a) {    auto b = a;    vi ind = iota(0, sz(a));    sortp(b, ind);    return ind;}
//#define use_sort
#ifdef use_sort
enum pcomparator { fisi, fisd, fdsi, fdsd, sifi, sifd, sdfi, sdfd };
enum tcomparator {    fisiti,    fisitd,    fisdti,    fisdtd,    fdsiti,    fdsitd,    fdsdti,    fdsdtd,    fitisi,    fitisd,    fitdsi,    fitdsd,    fdtisi,    fdtisd,    fdtdsi,    fdtdsd,    sifiti,    sifitd,    sifdti,    sifdtd,    sdfiti,    sdfitd,    sdfdti,    sdfdtd,    sitifi,    sitifd,    sitdfi,    sitdfd,    sdtifi,    sdtifd,    sdtdfi,    sdfdfd,    tifisi,    tifisd,    tifdsi,    tifdsd,    tdfisi,    tdfisd,    tdfdsi,    tdfdsd,    tisifi,    tisifd,    tisdfi,    tisdfd,    tdsifi,    tdsifd,    tdsdfi,    tdsdfd};
template<class A, class B> void sort(vector<pair<A, B>> &a, pcomparator type) {    typedef pair<A, B> U;    if (type == fisi) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi < r.fi : l.se < r.se; }); else if (type == fisd) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi < r.fi : l.se > r.se; }); else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi > r.fi : l.se < r.se; }); else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi > r.fi : l.se > r.se; }); else if (type == sifi) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se < r.se : l.fi < r.fi; }); else if (type == sifd) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se < r.se : l.fi > r.fi; }); else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se > r.se : l.fi < r.fi; }); else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se > r.se : l.fi > r.fi; });};
template<class U> void sort(vector<U> &a, pcomparator type) { if (type == fisi) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s < r.s; }); else if (type == fisd) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s > r.s; }); else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s < r.s; }); else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s > r.s; }); else if (type == sifi) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f < r.f; }); else if (type == sifd) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f > r.f; }); else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f < r.f; }); else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f > r.f; }); };
template<class A, class B, class C, class D> void sort(vector<F2<A, B, C, D> > &a, pcomparator type) {    typedef F2<A, B, C, D> U;    if (type == fisi) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b < r.b; }); else if (type == fisd) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b > r.b; }); else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b < r.b; }); else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b > r.b; }); else if (type == sifi) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a < r.a; }); else if (type == sifd) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a > r.a; }); else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a < r.a; }); else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a > r.a; });};
template<class U> void sort(vector<U> &a, tcomparator type) {    if (type == 0) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s < r.s : l.t < r.t; });    else if (type == 1) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s < r.s : l.t > r.t; });    else if (type == 2) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s > r.s : l.t < r.t; });    else if (type == 3) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s > r.s : l.t > r.t; });    else if (type == 4) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s < r.s : l.t < r.t; });    else if (type == 5) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s < r.s : l.t > r.t; });    else if (type == 6) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s > r.s : l.t < r.t; });    else if (type == 7) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s > r.s : l.t > r.t; });    else if (type == 8) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t < r.t : l.s < r.s; });    else if (type == 9) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t < r.t : l.s > r.s; });    else if (type == 10) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t > r.t : l.s < r.s; });    else if (type == 11) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t > r.t : l.s > r.s; });    else if (type == 12) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t < r.t : l.s < r.s; });    else if (type == 13) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t < r.t : l.s > r.s; });    else if (type == 14) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t > r.t : l.s < r.s; });    else if (type == 15) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t > r.t : l.s > r.s; });    else if (type == 16) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f < r.f : l.t < r.t; });    else if (type == 17) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f < r.f : l.t > r.t; });    else if (type == 18) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f > r.f : l.t < r.t; });    else if (type == 19) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f > r.f : l.t > r.t; });    else if (type == 20) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f < r.f : l.t < r.t; });    else if (type == 21) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f < r.f : l.t > r.t; });    else if (type == 22) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f > r.f : l.t < r.t; });    else if (type == 23) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f > r.f : l.t > r.t; });    else if (type == 24) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t < r.t : l.f < r.f; });    else if (type == 25) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t < r.t : l.f > r.f; });    else if (type == 26) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t > r.t : l.f < r.f; });    else if (type == 27) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t > r.t : l.f > r.f; });    else if (type == 28) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t < r.t : l.f < r.f; });    else if (type == 29) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t < r.t : l.f > r.f; });    else if (type == 30) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t > r.t : l.f < r.f; });    else if (type == 31) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t > r.t : l.f > r.f; });    else if (type == 32) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f < r.f : l.s < r.s; });    else if (type == 33) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f < r.f : l.s > r.s; });    else if (type == 34) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f > r.f : l.s < r.s; });    else if (type == 35) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f > r.f : l.s > r.s; });    else if (type == 36) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f < r.f : l.s < r.s; });    else if (type == 37) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f < r.f : l.s > r.s; });    else if (type == 38) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f > r.f : l.s < r.s; });    else if (type == 39) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f > r.f : l.s > r.s; });    else if (type == 40) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s < r.s : l.f < r.f; }); else if (type == 41) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s < r.s : l.f > r.f; }); else if (type == 42) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s > r.s : l.f < r.f; }); else if (type == 43) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s > r.s : l.f > r.f; }); else if (type == 44) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s < r.s : l.f < r.f; }); else if (type == 45) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s < r.s : l.f > r.f; }); else if (type == 46) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s > r.s : l.f < r.f; }); else if (type == 47) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s > r.s : l.f > r.f; });}
template<class A, class B, class C, class D> void sort(vector<F2<A, B, C, D>> &a, tcomparator type) {    typedef F2<A, B, C, D> U;    if (type == 0) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b < r.b : l.c < r.c; });    else if (type == 1) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b < r.b : l.c > r.c; });    else if (type == 2) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b > r.b : l.c < r.c; });    else if (type == 3) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b > r.b : l.c > r.c; });    else if (type == 4) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b < r.b : l.c < r.c; });    else if (type == 5) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b < r.b : l.c > r.c; });    else if (type == 6) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b > r.b : l.c < r.c; });    else if (type == 7) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b > r.b : l.c > r.c; });    else if (type == 8) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c < r.c : l.b < r.b; });    else if (type == 9) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c < r.c : l.b > r.b; });    else if (type == 10) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c > r.c : l.b < r.b; });    else if (type == 11) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c > r.c : l.b > r.b; });    else if (type == 12) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c < r.c : l.b < r.b; });    else if (type == 13) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c < r.c : l.b > r.b; });    else if (type == 14) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c > r.c : l.b < r.b; });    else if (type == 15) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c > r.c : l.b > r.b; });    else if (type == 16) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a < r.a : l.c < r.c; });    else if (type == 17) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a < r.a : l.c > r.c; });    else if (type == 18) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a > r.a : l.c < r.c; });    else if (type == 19) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a > r.a : l.c > r.c; });    else if (type == 20) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a < r.a : l.c < r.c; });    else if (type == 21) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a < r.a : l.c > r.c; });    else if (type == 22) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a > r.a : l.c < r.c; });    else if (type == 23) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a > r.a : l.c > r.c; });    else if (type == 24) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c < r.c : l.a < r.a; });    else if (type == 25) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c < r.c : l.a > r.a; });    else if (type == 26) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c > r.c : l.a < r.a; });    else if (type == 27) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c > r.c : l.a > r.a; });    else if (type == 28) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c < r.c : l.a < r.a; });    else if (type == 29) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c < r.c : l.a > r.a; });    else if (type == 30) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c > r.c : l.a < r.a; });    else if (type == 31) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c > r.c : l.a > r.a; });    else if (type == 32) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a < r.a : l.b < r.b; });    else if (type == 33) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a < r.a : l.b > r.b; });    else if (type == 34) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a > r.a : l.b < r.b; });    else if (type == 35) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a > r.a : l.b > r.b; });    else if (type == 36) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a < r.a : l.b < r.b; });    else if (type == 37) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a < r.a : l.b > r.b; });    else if (type == 38) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a > r.a : l.b < r.b; });    else if (type == 39) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a > r.a : l.b > r.b; });    else if (type == 40) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b < r.b : l.a < r.a; }); else if (type == 41) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b < r.b : l.a > r.a; }); else if (type == 42) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b > r.b : l.a < r.a; }); else if (type == 43) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b > r.b : l.a > r.a; }); else if (type == 44) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b < r.b : l.a < r.a; }); else if (type == 45) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b < r.b : l.a > r.a; }); else if (type == 46) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b > r.b : l.a < r.a; }); else if (type == 47) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b > r.b : l.a > r.a; });}

void sort(string &a) { sort(all(a)); }
void sort(int &a, int &b) { if (a > b)swap(a, b); }
void sort(int &a, int &b, int &c) {    sort(a, b);    sort(a, c);    sort(b, c);}
void rsort(int &a, int &b) { if (a < b)swap(a, b); }
void rsort(int &a, int &b, int &c) {    rsort(a, b);    rsort(a, c);    rsort(b, c);}
//P l, P rで f(P) の形で渡す
template<class U, class F> void sort(vector<U> &a, F f) { sort(all(a), [&](U l, U r) { return f(l) < f(r); }); };
template<class U, class F> void rsort(vector<U> &a, F f) { sort(all(a), [&](U l, U r) { return f(l) > f(r); }); };

//F = T<T>
//例えばreturn p.fi + p.se;
template<class A, class B, class F> void sortp(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    sort(c, f);    rep(i, sz(a)) a[i] = c[i].fi, b[i] = c[i].se;}
template<class A, class B, class F> void rsortp(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    rsort(c, f);    rep(i, sz(a))a[i] = c[i].first, b[i] = c[i].second;}
template<class A, class B, class C, class F> void sortt(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    sort(d, f);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C, class F> void rsortt(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    rsort(d, f);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C, class D> void sortf(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    auto e = vtof(a, b, c, d);    sort(e);    rep(i, sz(a)) a[i] = e[i].a, b[i] = e[i].b, c[i] = e[i].c, d[i] = e[i].d;}
template<class A, class B, class C, class D> void rsortf(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    auto e = vtof(a, b, c, d);    rsort(e);    rep(i, sz(a)) a[i] = e[i].a, b[i] = e[i].b, c[i] = e[i].c, d[i] = e[i].d;}
/*indexの分で型が変わるためpcomparatorが必要*/
template<class T> vi sorti(vector<T> &a, pcomparator f) {    auto b = a;    vi ind = iota(0, sz(a));    sortp(b, ind, f);    return ind;}
template<class T, class F> vi sorti(vector<T> &a, F f) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(a[x]) < f(a[y]); });    return ind;}
template<class T> vi rsorti(vector<T> &a) {    auto b = a;    vi ind = iota(0, sz(a));    rsortp(b, ind);    return ind;}
template<class T, class F> vi rsorti(vector<T> &a, F f) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(a[x]) > f(a[y]); });    return ind;}
template<class A, class B, class F> vi sortpi(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(c[x]) < f(c[y]); });    return ind;}
template<class A, class B> vi sortpi(vector<A> &a, vector<B> &b, pcomparator f) {    vi ind = iota(0, sz(a));    auto c = a;    auto d = b;    sortt(c, d, ind, f);    return ind;}
template<class A, class B> vi sortpi(vector<A> &a, vector<B> &b) { return sortpi(a, b, fisi); };
template<class A, class B, class F> vi rsortpi(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(c[x]) > f(c[y]); });    return ind;}
template<class A, class B> vi rsortpi(vector<A> &a, vector<B> &b) { return sortpi(a, b, fdsd); };
template<class A, class B, class C, class F> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(d[x]) < f(d[y]); });    return ind;}
template<class A, class B, class C> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c, pcomparator f) {    vi ind = iota(0, sz(a));    auto d = vtof(a, b, c, ind);    sort(d, f);    rep(i, sz(a))ind[i] = d[i].d;    return ind;}
template<class A, class B, class C> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { if (a[x] == a[y]) { if (b[x] == b[y])return c[x] < c[y]; else return b[x] < b[y]; } else { return a[x] < a[y]; }});    return ind;}
template<class A, class B, class C, class F> vi rsortti(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(d[x]) > f(d[y]); });    return ind;}
template<class A, class B, class C> vi rsortti(vector<A> &a, vector<B> &b, vector<C> &c) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { if (a[x] == a[y]) { if (b[x] == b[y])return c[x] > c[y]; else return b[x] > b[y]; } else { return a[x] > a[y]; }});    return ind;}
template<class T> void sort2(vector<vector<T >> &a) { for (ll i = 0, n = a.size(); i < n; ++i)sort(a[i]); }
template<class T> void rsort2(vector<vector<T >> &a) { for (ll i = 0, n = a.size(); i < n; ++i)rsort(a[i]); }

#endif
template<typename W, typename T> void fill(W &xx, const T vall) { xx = vall; }
template<typename W, typename T> void fill(vector<W> &vecc, const T vall) { for (auto &&vx     : vecc)fill(vx, vall); }
template<typename W, typename T> void fill(vector<W> &xx, const T v, ll len) { rep(i, len)xx[i] = v; }
template<typename W, typename T> void fill(vector<W> &xx,const T v, int s, ll t) { rep(i, s, t)xx[i] = v; }
template<typename W, typename T> void fill(vector<vector<W>> &xx, T v, int sh, int th, int sw, int tw) { rep(h, sh, th)rep(w, sw, tw)xx[h][w] = v; }
//#define use_fill //_sum _array _max _min
#ifdef use_fill
template<typename A, size_t N, typename T> void fill(A (&a)[N], const T &v) { rep(i, N)a[i] = v; }
template<typename A, size_t N, size_t O, typename T> void fill(A (&a)[N][O], const T &v) { rep(i, N)rep(j, O)a[i][j] = v; }
template<typename A, size_t N, size_t O, size_t P, typename T> void fill(A (&a)[N][O][P], const T &v) { rep(i, N)rep(j, O)rep(k, P)a[i][j][k] = v; }
template<typename A, size_t N, size_t O, size_t P, size_t Q, typename T> void fill(A (&a)[N][O][P][Q], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)a[i][j][k][l] = v; }
template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, typename T> void fill(A (&a)[N][O][P][Q][R], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)a[i][j][k][l][m] = v; }
template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S, typename T> void fill(A (&a)[N][O][P][Q][R][S], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)rep(n, S)a[i][j][k][l][m][n] = v; }
template<class T, class U> void fill(vector<T> &a, const vi &ind, U val) { fora_f(v, ind){fora_f_init(v, ind); a[v] = val;} }
template<typename A, size_t N> A sum(A (&a)[N]) {    A res = 0;    rep(i, N)res += a[i];    return res;}
template<typename A, size_t N, size_t O> A sum(A (&a)[N][O]) {    A res = 0;    rep(i, N)rep(j, O)res += a[i][j];    return res;}
template<typename A, size_t N, size_t O, size_t P> A sum(A (&a)[N][O][P]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)res += a[i][j][k];    return res;}
template<typename A, size_t N, size_t O, size_t P, size_t Q> A sum(A (&a)[N][O][P][Q]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)res += a[i][j][k][l];    return res;}
template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R> A sum(A (&a)[N][O][P][Q][R]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)res += a[i][j][k][l][m];    return res;}
template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S> A sum(A (&a)[N][O][P][Q][R][S]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)rep(n, S)res += a[i][j][k][l][m][n];    return res;}

template<typename A, size_t N> A max(A (&a)[N]) {    A res = a[0];    rep(i, N)res = max(res, a[i]);    return res;}template<typename A, size_t N, size_t O> A max(A (&a)[N][O]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P> A max(A (&a)[N][O][P]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q> A max(A (&a)[N][O][P][Q], const T &v) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R> A max(A (&a)[N][O][P][Q][R]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S> A max(A (&a)[N][O][P][Q][R][S]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}
template<typename A, size_t N> A min(A (&a)[N]) {    A res = a[0];    rep(i, N)res = min(res, a[i]);    return res;}template<typename A, size_t N, size_t O> A min(A (&a)[N][O]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P> A min(A (&a)[N][O][P]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q> A min(A (&a)[N][O][P][Q], const T &v) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R> A min(A (&a)[N][O][P][Q][R]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S> A min(A (&a)[N][O][P][Q][R][S]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}
#endif
template<class T, class U> void inc(pair<T, U> &a, U v = 1) { a.first += v, a.second += v; }
template<class T, class U> void inc(T &a, U v = 1) { a += v; }
template<class T, class U = int> void inc(vector<T> &a, U v = 1) { for (auto &u:a)inc(u, v); }
template<class T, class U> void dec(T &a, U v = 1) { a -= v; }
template<class T, class U = int> void dec(vector<T> &a, U v = 1) { for (auto &u :a)dec(u, v); }
template<class U> void dec(string &a, U v = 1) { for (auto &u :a)dec(u, v); }
template<class T, class U, class W> void dec(vector<T> &a, vector<U> &b, W v = 1) {    for (auto &u :a)dec(u, v);    for (auto &u :b)dec(u, v);}
template<class T, class U, class W> void dec(vector<T> &a, vector<U> &b, vector<W> &c) {    for (auto &u :a)dec(u, 1);    for (auto &u :b)dec(u, 1);    for (auto &u :c)dec(u, 1);}
bool ins(ll h, ll w, ll H, ll W) { return h >= 0 && w >= 0 && h < H && w < W; }
bool san(ll l, ll v, ll r) { return l <= v && v < r; }
template<class T> bool ins(vector<T> &a, ll i, ll j = 0) { return san(0, i, sz(a)) && san(0, j, sz(a)); }
#define inside ins
ll u0(ll a) { return a < 0 ? 0 : a; }
template<class T> vector<T> u0(vector<T> &a) {    vector<T> ret = a;    fora_f(v, ret){fora_f_init(v, ret); v = u(v); }   return ret;}
//todo 名前
bool d_(int a, int b) {    if (b == 0)return false;    return (a % b) == 0;}


//@汎用便利関数 入力
ll in() {    ll ret;    cin >> ret;    return ret;}
template<class T> T in() {    T ret;    cin >> ret;    return ret;}
string sin() {    string ret;    cin >> ret;    return ret;}
template<class T> void in(T &head) { cin >> head; }
template<class T, class... U> void in(T &head, U &... tail) {    cin >> head;    in(tail...);}
template<class T> T tin(int len = 0){    T a(len);    cin>>a;    return a;}
template<class T> T tind(int len = 0){    auto ret=tin<T>(len); dec(ret, 1); return ret;}
#define din_t2(type, a) type a;cin>>a
#define din_t3(type, a, b) type a,b;cin>>a>> b
#define din_t4(type, a, b, c) type a,b,c;cin>>a>>b>>c
#define din_t5(type, a, b, c, d) type a,b,c,d;cin>>a>>b>>c>>d
#define din_t6(type, a, b, c, d, e) type a,b,c,d,e;cin>>a>>b>>c>>d>>e
#define din_t7(type, a, b, c, d, e, f) type a,b,c,d,e,f;cin>>a>>b>>c>>d>>e>>f
#define din_t(...) over7(__VA_ARGS__,din_t7,din_t6,din_t5,din_t4,din_t3 ,din_t2)(__VA_ARGS__)

#define din(...) din_t(int,__VA_ARGS__)
#define d_in
#define dsig(...) din_t(signed,__VA_ARGS__)

#define dst(...) din_t(string,__VA_ARGS__)
#define dstr dst
#define d_str dst

#define dcha(...) din_t(char,__VA_ARGS__)
#define dchar dcha

#define ddou(...) din_t(double,__VA_ARGS__)


#define din1d(a) din_t2(int, a);a--
#define din2d(a, b) din_t3(int, a,b);a--,b--
#define din3d(a, b, c) din_t4(int, a,b,c);a--,b--,c--
#define din4d(a, b, c, d) din_t5(int, a,b,c,d);a--,b--,c--,d--
#define dind(...) over4(__VA_ARGS__,din4d,din3d,din2d ,din1d)(__VA_ARGS__)


#ifdef _DEBUG
template<class T> void err2(T &&head) { cerr << head; }
template<class T, class... U> void err2(T &&head, U &&... tail) {    cerr << head << " ";    err2(tail...);}
template<class T, class... U> void err(T &&head, U &&... tail) {    cerr << head << " ";    err2(tail...);    cerr << "" << endl;}
template<class T> void err(T &&head) { cerr << head << endl; }
void err() { cerr << "" << endl; }
template<class T> string out_m2(vector<T> &a, ll W = inf) {    stringstream ss;    if (W == inf)W = min(sz(a), 20ll);    if (sz(a) == 0)return ss.str();    rep(i, W) {        ss << a[i];        if (typeid(a[i]) == typeid(P)) { ss << endl; } else { ss << " "; }    }    return ss.str();}
template<class T> string out_m2(vector<vector<T> > &a, vi H, vi W, int key = -1) {    stringstream ss;    ss << endl;    vi lens(sz(W));    fora_f(h, H) {   fora_f_init(h, H);      rep(wi, sz(W)) {            int v = a[h][W[wi]];            str s = tos(v);            if (abs(v) == inf || abs(v) == INF<T>())s = "e";            lens[wi] = max(lens[wi], sz(s) + 1);            lens[wi] = max(lens[wi], sz(tos(W[wi])) + 1);        }    }    if (key == -1)ss << " *|"; else ss << " " << key << "|";    int wi = 0;    fora_f(w, W) {      fora_f_init(w, W);   ss << std::right << std::setw(lens[wi]) << w;        wi++;    }    ss << "" << endl;    rep(i, sz(W))rep(lens[i])ss << "_";    rep(i, 3)ss << "_";    ss << "" << endl;    fora_f(h, H) {      fora_f_init(h, H);   ss << std::right << std::setw(2) << h << "|";        int wi = 0;        fora_f(w, W) {      fora_f_init(w, W);       str s = tos(a[h][w]);            if (abs(a[h][w]) == inf || abs(a[h][w]) == INF<T>())s = "e";            ss << std::right << std::setw(lens[wi]) << s;            wi++;        }        ss << "" << endl;    }    return ss.str();}
template<class T> string out_m2(vector<vector<T> > &a, ll H = inf, ll W = inf, int key = -1) {    H = (H != inf) ? H : min({H, sz(a), 12ll});    W = min({W, sz(a[0]), 12ll});    vi hs, ws;    rep(h, H) { hs.push_back(h); }    rep(w, W) { ws.push_back(w); }    return out_m2(a, hs, ws, key);}
template<class T> string out_m2(vector<set<T> > &a, ll H = inf, ll W = inf, int key = -1) {    vector<vector<T> > b(sz(a));    rep(i, sz(a)) { fora_f(v, a[i]) { fora_f_init(v, a[i]); b[i].push_back(v); }}    return out_m2(b, H, W, key);}
template<class T> string out_m2(vector<vector<vector<T> > > &a, ll H = inf, ll W = inf, ll U = inf) {    stringstream ss;    if (H == inf)H = 12;    H = min(H, sz(a));    rep(i, H) {        ss << endl;        ss << out_m2(a[i], W, U, i);    }    return ss.str();}
//template<size_t A>string out_m2(T (&a)[A]){return out_m2(vector<T>(begin(a), end(a)));}
//template<size_t A, size_t B>string out_m2(T (&a)[A][B]){return out_m2(vector<vector<T > >(begin(a), end(a)));}
//template<size_t A, size_t B, size_t C>string out_m2(T (&a)[A][B][C]){return out_m2(vector<vector<vector<T> > >(begin(a), end(a)));}
template<class T, size_t N> string out_m2(T (&a)[N]) {    vector<T> b;    resize(b, N);    rep(i, N) { b[i] = a[i]; }    return out_m2(b);}
template<class T, size_t N, size_t M> string out_m2(T (&a)[N][M]) {    vector<vector<T>> b;    resize(b, N, M);    rep(i, N) { rep(j, M) { b[i][j] = a[i][j]; }}    return out_m2(b);}
template<class T, size_t N, size_t M, size_t O> string out_m2(T (&a)[N][M][O]) {    vector<vector<vector<T>>> b;    resize(b, N, M, O);    rep(i, N) { rep(j, M) { rep(k, O) { b[i][j][k] = a[i][j][k]; }}}    return out_m2(b);}
string out_m2(int a) {    stringstream ss;    ss << a;    return ss.str();}
template<class T> string out_m2(T &a) {    stringstream ss;    ss << a;    return ss.str();}
template<class T> string out_m(vector<T> &a, ll W = inf) {    stringstream ss;    if (W == inf)W = min(sz(a), 12ll);    if (sz(a) == 0)return ss.str();    rep(i, W) { ss << a[i] << " "; }    ss << "" << endl;    return ss.str();}
template<class T> string out_m(vector<vector<T> > &a, ll H = inf, ll W = inf, int key = -1) {    H = min({H, sz(a), 12ll});    W = min({W, sz(a[0]), 12ll});    stringstream ss;    ss << endl;    if (key == -1)ss << " *|"; else ss << " " << key << "|";    rep(w, W)ss << std::right << std::setw(4) << w;    ss << "" << endl;    rep(w, W * 4 + 3)ss << "_";    ss << "" << endl;    rep(h, H) {        ss << std::right << std::setw(2) << h << "|";        rep(w, min(sz(a[h]), 12ll)) { if (abs(a[h][w]) == INF<T>()) ss << "   e" << ""; else ss << std::right << std::setw(4) << a[h][w]; }        ss << "" << endl;    }    ss << endl;    return ss.str();}
template<class T> string out_m(vector<vector<vector<T> > > &a, ll H = inf, ll W = inf, ll U = inf) {    stringstream ss;    if (H == inf)H = 5;    H = min(H, sz(a));    rep(i, H) {        ss << endl;        ss << out_m(a[i], W, U, i);    }    ss << endl;    return ss.str();}
string out_m(int a) {    stringstream ss;    ss << a << endl;    return ss.str();}
template<class T> string out_m(T &a) {    stringstream ss;    ss << a << endl;    return ss.str();}
template<class T> void outv(vector<T> &a, ll W = inf) { cout << out_m(a, W) << endl; }
template<class T> void outv(vector<vector<T> > &a, ll H = linf, ll W = linf, int key = -1) { cout << out_m(a, H, W, key) << endl; }
template<class T> void outv(vector<vector<vector<T> > > &a, ll H = linf, ll W = linf, ll U = linf) { cout << out_m(a, H, W, U) << endl; }

template<class T> void out2(T head) {    cout << head;    res_mes += out_m2(head);}
template<class T, class... U> void out2(T head, U ... tail) {    cout << head << " ";    res_mes += out_m2(head) + " ";    out2(tail...);}
template<class T, class... U> void out(T head, U ... tail) {    cout << head << " ";    res_mes += out_m2(head) + " ";    out2(tail...);    cout << "" << endl;    res_mes += "\n";}
template<class T> void out(T head) {    cout << head << endl;    res_mes += out_m2(head) + "\n";}
void out() { cout << "" << endl; }

#else
template<class T> void outv(vector<T> &a, ll W = inf) {    rep(i, min(W, sz(a))) { cout << a[i] << " "; }    cout << "" << endl;}
template<class T> void outv(vector<vector<T> > &a, ll H = linf, ll W = linf, int key = -1) { rep(i, min(H, sz(a))) { outv(a[i], W); }}
template<class T> void outv(vector<vector<vector<T> > > &a, ll H = linf, ll W = linf, ll U = linf) { ; }
#define err(...);

template<class T> void out2(T &&head) { cout << head; }
template<class T, class... U> void out2(T &&head, U &&... tail) {    cout << head << " ";    out2(tail...);}
template<class T, class... U> void out(T &&head, U &&... tail) {    cout << head << " ";    out2(tail...);    cout << "" << endl;}
template<class T> void out(T &&head) {    cout << head  << endl;}
void out() {    cout << ""  << endl;}

#endif



template<class T> void outl(const vector<T> &a, int n = inf) { rep(i, min(n, sz(a)))cout << a[i] << endl; }
//テーブルをスペースなしで出力
template<class T> void outt(vector<vector<T>> &a) {rep(i, sz(a)) {rep(j, sz(a[i])) { cout << a[i][j]; }cout << endl;}}
//int型をbit表記で出力
void outb(int a) { cout << bitset<20>(a) << endl; }
template<class T> void na(vector<T> &a, ll n) {a.resize(n);rep(i, n)cin >> a[i];}
template<class T> void na(set<T> &a, ll n) { rep(i, n)a.insert(in()); }
#define dna(a, n) vi a; na(a, n);/*nを複数使うと n==in()の時バグる事に注意*/
#define dnad(a, n) vi a; nad(a, n);
template<class T> void nao(vector<T> &a, ll n) {a.resize(n + 1);a[0] = 0;rep(i, n)cin >> a[i + 1];}
template<class T> void naod(vector<T> &a, ll n) {a.resize(n + 1);a[0] = 0;rep(i, n)cin >> a[i + 1], a[i + 1]--;}
template<class T> void nad(vector<T> &a, ll n) {a.resize(n);rep(i, n)cin >> a[i], a[i]--;}
template<class T> void nad(set<T> &a, ll n) { rep(i, n)a.insert(in() - 1); }
template<class T, class U> void na2(vector<T> &a, vector<U> &b, ll n) {a.resize(n);b.resize(n);rep(i, n)cin >> a[i] >> b[i];}
template<class T, class U> void na2(set<T> &a, set<U> &b, ll n) {rep(i, n) {a.insert(in());b.insert(in());}}
#define dna2(a, b, n) vi a,b; na2(a,b,n);
template<class T, class U> void nao2(vector<T> &a, vector<U> &b, ll n) {a.resize(n + 1);b.resize(n + 1);a[0] = b[0] = 0;rep(i, n)cin >> a[i + 1] >> b[i + 1];}
template<class T, class U> void na2d(vector<T> &a, vector<U> &b, ll n) {a.resize(n);b.resize(n);rep(i, n)cin >> a[i] >> b[i], a[i]--, b[i]--;}
#define dna2d(a, b, n) vi a,b; na2d(a,b,n);
template<class T, class U, class W> void na3(vector<T> &a, vector<U> &b, vector<W> &c, ll n) {a.resize(n);b.resize(n);c.resize(n);rep(i, n)cin >> a[i] >> b[i] >> c[i];}
#define dna3(a, b, c, n) vi a,b,c; na3(a,b,c,n);
template<class T, class U, class W> void na3d(vector<T> &a, vector<U> &b, vector<W> &c, ll n) {a.resize(n);b.resize(n);c.resize(n);rep(i, n)cin >> a[i] >> b[i] >> c[i], a[i]--, b[i]--, c[i]--;}
#define dna3d(a, b, c, n) vi a,b,c; na3d(a,b,c,n);
template<class T, class U, class W, class X> void na4(vector<T> &a, vector<U> &b, vector<W> &c, vector<X> &d, ll n) {a.resize(n);b.resize(n);c.resize(n);d.resize(n);rep(i, n)cin >> a[i] >> b[i] >> c[i] >> d[i];}
#define dna4(a, b, c, d, n)  vi a,b,c,d; na4(a,b,c,d,n);
#define dna4d(a, b, c, d, n)  vi a,b,c,d; na4d(a,b,c,d,n);
#define nt(a, h, w) resize(a,h,w);rep(nthi,h)rep(ntwi,w) cin >> a[nthi][ntwi];
#define ntd(a, h, w) resize(a,h,w);rep(ntdhi,h)rep(ntdwi,w) cin >> a[ntdhi][ntdwi], a[ntdhi][ntdwi]--;
#define ntp(a, h, w) resize(a,h+2,w+2);fill(a,'#');rep(ntphi,1,h+1)rep(ntpwi,1,w+1) cin >> a[ntphi][ntpwi];
#define dnt(S, h, w) vvi(S,h,w);nt(S,h,w);
#define dntc(S, h, w) vvc(S,h,w);nt(S,h,w);
#define dnts(S, h, w) vvs(S,h,w);nt(S,h,w);

//デバッグ
#define sp << " " <<

#define deb1(x)  debugName(x)<<" = "<<out_m2(x)
#define deb_2(x, ...) deb1(x) <<", "<< deb1(__VA_ARGS__)
#define deb_3(x, ...) deb1(x) <<", "<< deb_2(__VA_ARGS__)
#define deb_4(x, ...) deb1(x) <<", "<< deb_3(__VA_ARGS__)
#define deb5(x, ...) deb1(x) <<", "<< deb_4(__VA_ARGS__)
#define deb6(x, ...) deb1(x) <<", "<< deb5(__VA_ARGS__)
#define deb7(x, ...) deb1(x) <<", "<< deb6(__VA_ARGS__)
#define deb8(x, ...) deb1(x) <<", "<< deb7(__VA_ARGS__)
#define deb9(x, ...) deb1(x) <<", "<< deb8(__VA_ARGS__)
#define deb10(x, ...) deb1(x) <<", "<< deb9(__VA_ARGS__)


#ifdef _DEBUG
bool was_deb = false;
#define deb(...)  do{was_deb=true;cerr<< over10(__VA_ARGS__,deb10,deb9,deb8,deb7,deb6,deb5,deb_4,deb_3,deb_2,deb1)(__VA_ARGS__) <<endl;}while(0)

#define base_keta 8
void print_n_base(int x, int base) { cerr << bitset<base_keta>(x) << endl; }
template<class T> void print_n_base(vector<T> X, int base) {cerr << endl;for (auto &&x:X) { print_n_base(x, base); }cerr << endl;}

//n進数
#define deb2(x) was_deb=true;cerr<<debugName(x)<<" = ";print_n_base(x, 2);
#define deb3(x) was_deb=true;cerr<<debugName(x)<<" = ";print_n_base(x, 3);
#define deb4(x) was_deb=true;cerr<<debugName(x)<<" = ";print_n_base(x, 4);

#define deb_ex_deb(x, len)  debugName(x)<<" = "<<out_m2(x, len)
#define call_deb_ex_deb(x, len) deb_ex_deb(x, len)
//要素が存在する行だけ出力(vvt)
#define deb_ex(v) do {int N = sz(v);int s = N;int t = 0;rep(i, N) {if (sz(v[i])) {chmi(s, i);chma(t, i);}}auto ex_v = sub(v, s, N);str S = out_m2(ex_v, sz(ex_v));debugName(v);cerr<<" = "<<endl;cerr << S << endl;} while (0);
template<class T, class F> string out_m2_f(vector<vector<T> > &a, F f, int key = -1) {    vi hs, ws_;    int H = sz(a), W = sz(a[0]);    vi exh(H), exw(W);    rep(h, H) {        rep(w, W) {            if (f(a[h][w])) {                exh[h] = true;                exw[w] = true;            }        }    }    rep(h, H) if (exh[h])hs.push_back(h);    rep(w, W) if (exw[w])ws_.push_back(w);    return out_m2(a, hs, ws_, key);}
template<class T, class F> string out_m2_f(vector<vector<vector<T>>> &a, F f) {    stringstream ss;    int H = sz(a);    if (sz(a) == 0)return ss.str();    rep(i, H) { ss << out_m2_f(a[i], f, i); }    ss << "" << endl;    return ss.str();}
#define debf_normal(tab, f) do{cerr<<debugName(tab)<<" = "<<endl;cerr<< out_m2_f(tab, f)<<endl;}while(0);
#define debf2(tab, siki_r) debf_normal(tab, lamr(siki_r))
#define debf3(tab, v, siki)  debf_normal(tab, lam(siki))
//S, sikir
//S, v, siki
#define debf(...) over3(__VA_ARGS__,debf3,debf2,debf1)(__VA_ARGS__)

#else
#define deb(...) ;
#define deb2(...) ;
#define deb3(...) ;
#define deb4(...) ;
#define deb_ex(...) ;
#define debf(...) ;
#endif


#define debugline(x) cerr << x << " " << "(L:" << __LINE__ << ")" << '\n'


using u32 = unsigned;
using u64 = unsigned long long;
using u128 = __uint128_t;

using bint =__int128;
std::ostream &operator<<(std::ostream &dest, __int128_t value) {    std::ostream::sentry s(dest);    if (s) {        __uint128_t tmp = value < 0 ? -value : value;        char buffer[128];        char *d = std::end(buffer);        do {            --d;            *d = "0123456789"[tmp % 10];            tmp /= 10;        }        while (tmp != 0);        if (value < 0) {            --d;            *d = '-';        }        ll len = std::end(buffer) - d;        if (dest.rdbuf()->sputn(d, len) != len) { dest.setstate(std::ios_base::badbit); }    }    return dest;}
__int128 to_bint(string &s) {    __int128 ret = 0;    for (ll i = 0; i < s.length(); ++i) if ('0' <= s[i] && s[i] <= '9') ret = 10 * ret + s[i] - '0';    return ret;}
void operator>>(istream &iss, bint &v) {    string S;    iss >> S;    v = 0;    rep(i, sz(S)) {        v *= 10;        v += S[i] - '0';    }}
//エラー
void ole() {
#ifdef _DEBUG
    debugline("ole");exit(0);
#endif
    string a = "a";rep(i, 30)a += a;rep(i, 1 << 17)cout << a << endl;cout << "OLE 出力長制限超過" << endl;exit(0);
}
void re(string s="") {cerr<<s<<endl;assert(0 == 1);exit(0);}
void tle() { while (inf)cout << inf << endl; }

//便利関数

//テスト用
#define rand xor128_
unsigned long xor128_(void) {    static unsigned long x = 123456789, y = 362436069, z = 521288629, w = 88675123;    unsigned long t;    t = (x ^ (x << 11));    x = y;    y = z;    z = w;    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));}
char ranc() { return (char) ('a' + rand() % 26); }
ll rand(ll min, ll max) {    assert(min <= max);    if (min >= 0 && max >= 0) { return rand() % (max + 1 - min) + min; } else if (max < 0) { return -rand(-max, -min); } else { if (rand() % 2) { return rand(0, max); } else { return -rand(0, -min); }}}
ll rand(ll max) { return rand(0, max); }
template<class T> T rand(vector<T> &A) { return A[rand(sz(A) - 1)]; }
//重複することがある
template<class T> vector<T> ranv(vector<T>& A, int N){vector<T>ret(N);rep(i,N){ret[i]=rand(A);}return ret;}
template<class T> vector<T> ranv_unique(vector<T>& A, int N){    vector<T> ret(N);    umapi was;    rep(j,N){        int i;        while(1){            i = rand(sz(A)-1);            if(was.find(i) == was.end())break;        }        ret[j] = A[i];was[i]=1;    }    return ret;}
vi ranv(ll n, ll min, ll max) {    vi v(n);    rep(i, n)v[i] = rand(min, max);    return v;}

#ifdef _DEBUG
bool timeup(int time){
    static bool never = true;
    if(never)message += "may timeup, because slow";
    never = false;

    auto end_time = system_clock::now();
    auto part = duration_cast<milliseconds>(end_time - start_time);
    auto lim = milliseconds(time);
    return part >= lim;
}
#else
bool timeup(int time) {
    auto end_time = system_clock::now();
    auto part = duration_cast<milliseconds>(end_time - start_time);
    auto lim = milliseconds(time);
    return part >= lim;
}

#endif
void set_time() { past_time = system_clock::now(); }
//MS型(millisecqnds)で返る
//set_timeをしてからの時間
auto calc_time_milli() {    auto now = system_clock::now();    auto part = duration_cast<milliseconds>(now - past_time);    return part;}
auto calc_time_micro() {    auto now = system_clock::now();    auto part = duration_cast<microseconds>(now - past_time);    return part;}
auto calc_time_nano() {    auto now = system_clock::now();    auto part = duration_cast<nanoseconds>(now - past_time);    return part;}
bool calc_time(int zikan) { return calc_time_micro() >= microseconds(zikan); }
using MS=std::chrono::microseconds;
int div(microseconds a, microseconds b) { return a / b; }
int div(nanoseconds a, nanoseconds b) {    if (b < nanoseconds(1)) { return a / nanoseconds(1); }    int v = a / b;    return v;}
//set_time();
//rep(i,lim)shori

//lim*=time_nanbai();
//rep(i,lim)shoriと使う
//全体でmilliかかっていいときにlimを何倍してもう一回できるかを返す
int time_nanbai(int milli) {
    auto dec = duration_cast<nanoseconds>(past_time - start_time);
    auto part = calc_time_nano();
    auto can_time = nanoseconds(milli * 1000 * 1000);
    can_time -= part;
    can_time -= dec;
    return div(can_time, part);
}

//#define use_rand
#ifdef use_rand
str ransu(ll n) {    str s;    rep(i, n)s += (char) rand('A', 'Z');    return s;}
str ransl(ll n) {    str s;    rep(i, n)s += (char) rand('a', 'z');    return s;}
//単調増加
vi ranvinc(ll n, ll min, ll max) {    vi v(n);    bool bad = 1;    while (bad) {        bad = 0;        v.resize(n);        rep(i, n) {            if (i && min > max - v[i - 1]) {                bad = 1;                break;            }            if (i)v[i] = v[i - 1] + rand(min, max - v[i - 1]); else v[i] = rand(min, max);        }    }    return v;}
//便利 汎用
#endif

void ranvlr(ll n, ll min, ll max, vi &l, vi &r) {    l.resize(n);    r.resize(n);    rep(i, n) {        l[i] = rand(min, max);        r[i] = l[i] + rand(0, max - l[i]);    }}
template<class T> vector<pair<T, int>> run_length(vector<T> &a) {    vector<pair<T, int>> ret;    ret.eb(a[0], 1);    rep(i, 1, sz(a)) { if (ret.back().fi == a[i]) { ret.back().se++; } else { ret.eb(a[i], 1); }}    return ret;}
vector<pair<char, ll>> run_length(const string &a) {    vector<pair<char, ll>> ret;    ret.eb(a[0], 1);    rep(i, 1, sz(a)) { if (ret.back().fi == a[i]) { ret.back().se++; } else { ret.eb(a[i], 1); }}    return ret;}
//#define use_mgr //_goldd _goldt
#ifdef use_mgr
template<class T, class F> T mgr(T ok, T ng, F f, int deb_ = 0) {    bool han = true;    if (deb_) {        if (ok < ng)            while (ng - ok > 1) {                T mid = (ok + ng) >> 1;                if (f(mid))ok = mid, han = true; else ng = mid, han = false;                deb(mid, han);            }        else            while (ok - ng > 1) {                T mid = (ok + ng) >> 1;                if (f(mid))ok = mid, han = true; else ng = mid, han = false;                deb(mid, han);            }    }    else {        if (ok < ng)            while (ng - ok > 1) {                T mid = (ok + ng) >> 1;                if (f(mid))ok = mid, han = true; else ng = mid, han = false;            }        else            while (ok - ng > 1) {                T mid = (ok + ng) >> 1;                if (f(mid))ok = mid, han = true; else ng = mid, han = false;            }    }    return ok;}
template<class T, class F> T mgr(signed ok, T ng, F f) { return mgr((T) ok, ng, f); }
template<class T, class F> T mgr(T ok, signed ng, F f) { return mgr(ok, (T) ng, f); }
template<class F> int mgr(signed ok, signed ng, F f) { return mgr((ll) ok, (ll) ng, f); }
//[l, r)の中で,f(i)がtrueとなる範囲を返す okはそこに含まれる
template<class F> P mgr(int l, int r, F f, int ok) {    if (f(ok) == 0) {        out("f(ok) must true");        re();    }    return mp(mgr(ok, l - 1, f), mgr(ok, r, f) + 1);}

template<class F> dou mgrd(dou ok, dou ng, F f, int kai = 100) {    bool han = true;    if (ok < ng)        rep(i, kai) {            dou mid = (ok + ng) / 2;            if (f(mid))ok = mid, han = true; else ng = mid, han = false;            deb(mid, han);        }    else        rep(i, kai) {            dou mid = (ok + ng) / 2;            if (f(mid))ok = mid, han = true; else ng = mid, han = false;            deb(mid, han);        }    return ok;}

template<class F> dou mgrd_time(dou ok, dou ng, F f,int time = 1980) {
    bool han = true;
    if (ok < ng)
        while(1) {
            dou mid = (ok + ng) /2;
            if (f(mid))ok = mid, han = true; else ng = mid, han = false;
            deb(mid,han);
            if(timeup(time)){
                break;
            }
        }
    else
        while(1) {
            dou mid = (ok + ng) /2;
            if (f(mid))ok = mid, han = true; else ng = mid, han = false;
            deb(mid,han);
            if(timeup(time)){
                break;
            }
        }
    return ok;
}


//添え字を返す
template<class F> ll goldd_l(ll left, ll right, F calc) {    double GRATIO = 1.6180339887498948482045868343656;    ll lm = left + (ll) ((right - left) / (GRATIO + 1.0));    ll rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));    ll fl = calc(lm);    ll fr = calc(rm);    while (right - left > 10) {        if (fl < fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + (ll) ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        }        else {            left = lm;            lm = rm;            fl = fr;            rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    ll minScore = MAX<ll>();    ll resIndex = left;    for (ll i = left; i < right + 1; ++i) {        ll score = calc(i);        if (minScore > score) {            minScore = score;            resIndex = i;        }    }    return resIndex;}
template<class F> ll goldt_l(ll left, ll right, F calc) {    double GRATIO = 1.6180339887498948482045868343656;    ll lm = left + (ll) ((right - left) / (GRATIO + 1.0));    ll rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));    ll fl = calc(lm);    ll fr = calc(rm);    while (right - left > 10) {        if (fl > fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + (ll) ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        }        else {            left = lm;            lm = rm;            fl = fr;            rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    if (left > right) {        ll l = left;        left = right;        right = l;    }    ll maxScore = MIN<ll>();    ll resIndex = left;    for (ll i = left; i < right + 1; ++i) {        ll score = calc(i);        if (maxScore < score) {            maxScore = score;            resIndex = i;        }    }    return resIndex;}
/*loopは200にすればおそらく大丈夫 余裕なら300に*/
template<class F> dou goldd_d(dou left, dou right, F calc, ll loop = 200) {    dou GRATIO = 1.6180339887498948482045868343656;    dou lm = left + ((right - left) / (GRATIO + 1.0));    dou rm = lm + ((right - lm) / (GRATIO + 1.0));    dou fl = calc(lm);    dou fr = calc(rm);    /*200にすればおそらく大丈夫*/    /*余裕なら300に*/    ll k = 141;    loop++;    while (--loop) {        if (fl < fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        }        else {            left = lm;            lm = rm;            fl = fr;            rm = lm + ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    return left;}
template<class F> dou goldt_d(dou left, dou right, F calc, ll loop = 200) {    double GRATIO = 1.6180339887498948482045868343656;    dou lm = left + ((right - left) / (GRATIO + 1.0));    dou rm = lm + ((right - lm) / (GRATIO + 1.0));    dou fl = calc(lm);    dou fr = calc(rm);    loop++;    while (--loop) {        if (fl > fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        }        else {            left = lm;            lm = rm;            fl = fr;            rm = lm + ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    return left;}
//l ~ rを複数の区間に分割し、極致を与えるiを返す time-20 msまで探索
template<class F> ll goldd_ls(ll l, ll r, F calc, ll time = 2000) {    auto lim = milliseconds(time - 20);    ll mini = 0, minv = MAX<ll>();    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        ll haba = (r - l + k) / k;/*((r-l+1) + k-1) /k*/        ll nl = l;        ll nr = l + haba;        rep(i, k) {            ll ni = goldd_l(nl, nr, calc);            if (chmi(minv, calc(ni))) mini = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return mini;}
template<class F> ll goldt_ls(ll l, ll r, F calc, ll time = 2000) {    auto lim = milliseconds(time - 20);    ll maxi = 0, maxv = MIN<ll>();    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        ll haba = (r - l + k) / k;/*((r-l+1) + k-1) /k*/        ll nl = l;        ll nr = l + haba;        rep(i, k) {            ll ni = goldt_l(nl, nr, calc);            if (chma(maxv, calc(ni))) maxi = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return maxi;}
template<class F> dou goldd_d_s(dou l, dou r, F calc, ll time = 2000) {    /*20ms余裕を持つ*/    auto lim = milliseconds(time - 20);    dou mini = 0, minv = MAX<dou>();    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        dou haba = (r - l) / k;        dou nl = l;        dou nr = l + haba;        rep(i, k) {            dou ni = goldd_d(nl, nr, calc);            if (chmi(minv, calc(ni))) mini = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return mini;}
template<class F> dou goldt_d_s(dou l, dou r, F calc, ll time = 2000) {    /*20ms余裕を残している*/    auto lim = milliseconds(time - 20);    dou maxi = 0, maxv = MIN<dou>();    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        dou haba = (r - l) / k;        dou nl = l;        dou nr = l + haba;        rep(i, k) {            dou ni = goldt_d(nl, nr, calc);            if (chma(maxv, calc(ni))) maxi = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return maxi;}

#endif

//strを整数として比較
string smax(str &a, str b) { if (sz(a) < sz(b)) { return b; } else if (sz(a) > sz(b)) { return a; } else if (a < b)return b; else return a; }
//strを整数として比較
string smin(str &a, str b) { if (sz(a) > sz(b)) { return b; } else if (sz(a) < sz(b)) { return a; } else if (a > b)return b; else return a; }
//エラー-1
template<typename W, typename T> ll find(vector<W> &a, int l, const T key) {rep(i, l, sz(a))if (a[i] == key)return i;return -1;}
template<typename W, typename T> ll find(vector<W> &a, const T key) {rep(i, sz(a))if (a[i] == key)return i;return -1;}
template<typename W, typename T> P find(vector<vector<W >> &a, const T key) {rep(i, sz(a))rep(j, sz(a[0]))if (a[i][j] == key)return mp(i, j);return mp(-1, -1);}
//getid(find())を返す 1次元にする
template<typename W, typename T> int findi(vector<vector<W >> &a, const T key) {rep(i, sz(a))rep(j, sz(a[0]))if (a[i][j] == key)return i * sz(a[0]) + j;return -1;}
template<typename W, typename U> T find(vector<vector<vector<W >>> &a, const U key) {rep(i, sz(a))rep(j, sz(a[0]))rep(k, sz(a[0][0]))if (a[i][j][k] == key)return mt(i, j, k);return mt(-1, -1, -1);}
//無ければ-1
int find(string &s, const string key) {    int klen = sz(key);    rep(i, sz(s) - klen + 1) {        if (s[i] != key[0])continue;        if (s.substr(i, klen) == key) { return i; }    }return -1;}
int find(string &s, int l, const string key) {    int klen = sz(key);    rep(i, l, sz(s) - klen + 1) {        if (s[i] != key[0])continue;        if (s.substr(i, klen) == key) { return i; }    }    return -1;}
int find(string &s, const char key) {    rep(i, sz(s)) { if (s[i] == key)return i; }    return -1;}
int find(string &s, int l, const char key) {    rep(i, l, sz(s)) { if (s[i] == key)return i; }    return -1;}
//N箇所について右のkeyの場所を返す
template<typename W, typename T> vi finds(const W &a, const T& key) {    int n = sz(a);    vi rpos(n, -1);    rer(i, n-1){        if(i<n-1){            rpos[i] = rpos[i+1];        }        if(a[i]==key)rpos[i] = i;    }    return rpos;}
template<typename W, typename T> vi rfinds(const W &a, const T& key) {    int n = sz(a);    vi lpos(n, -1);    rep(i, n){        if(i> 0){            lpos[i] = lpos[i-1];        }        if(a[i]==key)lpos[i] = i;    }    return lpos;}
template<typename W, typename T> ll count2(W &a, const T k) { return a == k; }
template<typename W, typename T> ll count2(vector<W> &a, const T k) {    ll ret = 0;    fora_f(v, a){fora_f_init(v, a);ret += count2(v, k);  }  return ret;}
template<typename W, typename T> ll count(vector<W> &a, const T k) {    ll ret = 0;    fora_f(v, a){fora_f_init(v, a)ret += count2(v, k);}    return ret;}
vi count(vi &a) {    int ma = 0;    fora_f(v, a) { fora_f_init(v, a);if (ma < v)ma = v; }    vi res(ma + 1);    fora_f(v, a) { fora_f_init(v, a)res[v]++; }    return res;}
ll count(str &a, str k) {    ll ret = 0, len = k.length();    auto pos = a.find(k);    while (pos != string::npos)pos = a.find(k, pos + len), ++ret;    return ret;}
ll count(const str &a, char k){    int cou=0;    for(auto && c :a ){        cou += c==k;    }    return cou;}
//'a' = 'A' = 0 として集計 既に-'a'されていても動く
vi count(str &a, int l, int r) {    vi cou(26);    char c = 'a';    if ('A' <= a[l] && a[l] <= 'Z')c = 'A';    if ('a' <= a[l] && a[l] <= 'z') c = 'a'; else c = 0;    rep(i, l, r)++cou[a[i] - c];    return cou;}
#define couif count_if
//algorythm
ll rev(ll a) {    ll res = 0;    while (a) {        res *= 10;        res += a % 10;        a /= 10;    }    return res;}
template<class T> auto rev(vector<T> &a) {    auto b = a;    reverse(all(b));    return b;}
/* ＼反転  */                                                                                        template<class U>
auto rev(vector<vector<U>> &a) {    vector<vector<U> > b(sz(a[0]), vector<U>(sz(a)));    rep(h, sz(a)) rep(w, sz(a[0]))b[w][h] = a[h][w];    return b;}
/* ｜反転  */                                                                                        template<class U>
auto revw(vector<vector<U>> &a) {    vector<vector<U> > b(sz(a), vector<U>(sz(a[0])));    int W = sz(a[0]);    rep(h, sz(a)) rep(w, sz(a[0])) { b[h][W - 1 - w] = a[h][w]; }    return b;}

/* ー反転  */                                                                                        template<class U>
auto revh(vector<vector<U>> &a) {    vector<vector<U> > b(sz(a), vector<U>(sz(a[0])));    int H = sz(a);    rep(h, sz(a)) rep(w, sz(a[0])) { b[H - 1 - h][w] = a[h][w]; }    return b;}

/* ／反転  */                                                                                        template<class U>
auto revr(vector<vector<U>> &a) {    vector<vector<U> > b(sz(a[0]), vector<U>(sz(a)));    int H = sz(a);    int W = sz(a[0]);    rep(h, sz(a)) rep(w, sz(a[0]))b[w][h] = a[H - 1 - h][W - 1 - w];    return b;}

auto rev(const string &a) {    string b = a;    reverse(all(b));    return b;}
constexpr ll p10[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000ll, 100000000000ll, 1000000000000ll, 10000000000000ll, 100000000000000ll, 1000000000000000ll, 10000000000000000ll, 100000000000000000ll, 1000000000000000000ll};

//0は0桁
ll keta(ll v, int if_zero_res) { if(!v)return if_zero_res;if (v < p10[9]) { if (v < p10[4]) { if (v < p10[2]) { if (v < p10[1]) { if (v < p10[0])return 0; else return 1; } else return 2; } else { if (v < p10[3]) return 3; else return 4; }} else { if (v < p10[7]) { if (v < p10[5]) return 5; else if (v < p10[6])return 6; else return 7; } else { if (v < p10[8])return 8; else return 9; }}} else { if (v < p10[13]) { if (v < p10[11]) { if (v < p10[10]) return 10; else return 11; } else { if (v < p10[12]) return 12; else return 13; }} else { if (v < p10[15]) { if (v < p10[14]) return 14; else return 15; } else { if (v < p10[17]) { if (v < p10[16]) return 16; else return 17; } else { if (v < p10[18])return 18; else return 19; }}}}}
ll getr(ll a, ll keta) { return (a / (ll) pow(10, keta)) % 10; }
//上から何桁目か
ll getl(ll a, ll ket) {int sketa = keta(a, 1);return getr(a, sketa - 1 - ket);}
ll dsum(ll v, ll sin = 10) {ll ret = 0;for (; v; v /= sin)ret += v % sin;return ret;}

ll mask10(ll v) { return p10[v] - 1; }
//変換系
//[v] := iとなるようなvectorを返す
//存在しない物は-1
//空でも動く(なぜか)
template<class T> auto keys(const T& a) {    vector<decltype((a.begin())->fi)> res;    for (auto &&k :a)res.push_back(k.fi);    return res;}
template<class T> auto values(const T& a) {    vector<decltype((a.begin())->se)> res;    for (auto &&k :a)res.push_back(k.se);    return res;}
template<class T> constexpr T min(T a, T b, T c) { return a >= b ? b >= c ? c : b : a >= c ? c : a; }
template<class T> constexpr T max(T a, T b, T c) { return a <= b ? b <= c ? c : b : a <= c ? c : a; }
template<class T> T min(vector<T> &a, ll s = -1, ll n = -1) { pr_set_lr(s, n, sz(a)); return *min_element(a.begin() + s, a.begin() + min(n, sz(a))); }
template<class T> T max(vector<T> &a, ll s = -1, ll n = -1) { pr_set_lr(s, n,sz(a)); return *max_element(a.begin() + s, a.begin() + min(n, sz(a))); }
template<class T> T mini(vector<T> &a) { return min_element(all(a)) - a.begin(); }
template<class T> T maxi(vector<T> &a) { return max_element(all(a)) - a.begin(); }

template<class T> T sum(const vector<T>& A, int l = -1, int r = -1){T s=0;pr_set_lr(l, r, sz(A));rep(i, l, r)s+=A[i];return s;}
template<class T> auto sum(const vector<vector<T>>& A ){decl2(A) s=0;rep(i, sz(A))s+=sum(A[i]);return s;}
template<class T> T min(const vector<T>& A, int l = -1, int r = -1 ){T s=MAX<T>();pr_set_lr(l, r, sz(A));rep(i, l, r)s=min(s, A[i]);return s;}
template<class T> auto min(const vector<vector<T>>& A ){using S =decl2(A);S s=MAX<S>();rep(i, sz(A))s=min(s, A[i]);return s;}
template<class T> T max(const vector<T>& A, int l = -1, int r = -1 ){T s=MIN<T>();pr_set_lr(l, r, sz(A));rep(i, l, r); rep(i, l, r)s=max(s, A[i]);return s;}
template<class T> auto max(const vector<vector<T>>& A ){using S =decl2(A);S s=MIN<S>();rep(i, sz(A))s=max(s, A[i]);return s;}
template<class T> T mul(vector<T> &v, ll t = inf) {    T ret = v[0];    rep(i, 1, min(t, sz(v)))ret *= v[i];    return ret;}

//template<class T, class U, class... W>  auto sumn(vector<T> &v, U head, W... tail) {    auto ret = sum(v[0], tail...);    rep(i, 1, min(sz(v), head))ret += sum(v[i], tail...);    return ret;}
//indexを持つvectorを返す
vi inds_(vi &a) {    int n = max(a) + 1;    vi ret(n, -1);    rep(i, sz(a)) { assert(ret[a[i]] ==-1);ret[a[i]] = i; }    return ret;}
void clear(PQ &q) { q = PQ(); }
void clear(priority_queue<int> &q) { q = priority_queue<int>(); }
template<class T> void clear(queue<T> &q) { while (q.size())q.pop(); }
//template<class T> T *negarr(ll size) {    T *body = (T *) malloc((size * 2 + 1) * sizeof(T));    return body + size;}
//template<class T> T *negarr2(ll h, ll w) {    double **dummy1 = new double *[2 * h + 1];    double *dummy2 = new double[(2 * h + 1) * (2 * w + 1)];    dummy1[0] = dummy2 + w;    for (ll i = 1; i <= 2 * h + 1; ++i) { dummy1[i] = dummy1[i - 1] + 2 * w + 1; }    double **a = dummy1 + h;    return a;}


template<class T> struct ruiC {
    vector<T> rui;
    ruiC(vector<T> &ru) : rui(ru) {}
    /*先頭0*/
    ruiC() : rui(1, 0) {}
    T operator()(ll l, ll r) {        if (l > r) {            cerr << "ruic ";            deb(l, r);            assert(0);        }        return rui[r] - rui[l];    }
    T operator()(int r = inf) { return operator()(0, min(r, sz(rui) - 1)); }
    /*ruiv[]をruic[]に変えた際意味が変わるのがまずいため()と統一*/
    /*単体iを返す 累積でないことに注意(seg木との統一でこうしている)*/
//    T operator[](ll i) { return rui[i + 1] - rui[i]; }
    T operator[](ll i) { return rui[i]; }
    /*0から順に追加される必要がある*/
    void operator+=(T v) { rui.push_back(rui.back() + v); }
    void add(int i, T v) {if (sz(rui) - 1 != i)ole();operator+=(v);}
    T back() { return rui.back(); }
    ll size() { return rui.size(); }
    auto begin() { return rui.begin(); }
    auto end() { return rui.end(); }
};
template<class T> ostream &operator<<(ostream &os, ruiC<T> a) {    fora_f(v, a.rui){fora_f_init(v, a.rui)os << v << " "; }   return os;}
template<class T> vector<T> ruiv(vector<T> &a) {    vector<T> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + a[i];    return ret;}
template<class T> ruiC<T> ruic(vector<T> &a) {    vector<T> ret = ruiv(a);    return ruiC<T>(ret);}
template<class T> ruiC<T> ruic() { return ruiC<T>(); }
//imoは0-indexed
//ruiは1-indexed
template<class T> vector<T> imo(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)ret[i + 1] += ret[i];    return ret;}

//#define use_rui //_imo _ruic _ruiv
#ifdef use_rui
//kと同じものの数
template<class T, class U> vi imo(const vector<T> &a, U k) {    vi equ(sz(a));    rep(i, sz(a)){        equ[i] = a[i]==k;    }    return imo(equ);}
template<class T> vector<T> imox(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)ret[i + 1] ^= ret[i];    return ret;}
//漸化的に最小を持つ
template<class T> vector<T> imi(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)chmi(ret[i + 1], ret[i]);    return ret;}
template<class T> vector<T> ima(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)chma(ret[i + 1], ret[i]);    return ret;}
template<class T> vector<T> rimi(vector<T> &v) {    vector<T> ret = v;    rer(i, sz(ret) - 1, 1)chmi(ret[i - 1], ret[i]);    return ret;}
template<class T> vector<T> rima(vector<T> &v) {    vector<T> ret = v;    rer(i, sz(ret) - 1, 1)chma(ret[i - 1], ret[i]);    return ret;}
template<class T> struct ruimax {
    template<typename Monoid> struct SegmentTree {        /*pairで処理*/        int sz;        vector<Monoid> seg;        const Monoid M1 = mp(MIN<T>(), -1);        Monoid f(Monoid a, Monoid b) { return max(a, b); }        void build(vector<T> &a) {            int n = sz(a);            sz = 1;            while (sz < n) sz <<= 1;            seg.assign(2 * sz, M1);            rep(i, n) { seg[i + sz] = mp(a[i], i); }            for (int k = sz - 1; k > 0; k--) { seg[k] = f(seg[k << 1], seg[(k << 1) | 1]); }        }        Monoid query(int a, int b) {            Monoid L = M1, R = M1;            for (a += sz, b += sz; a < b; a >>= 1, b >>= 1) {                if (a & 1) L = f(L, seg[a++]);                if (b & 1) R = f(seg[--b], R);            }            return f(L, R);        }        Monoid operator[](const int &k) const { return seg[k + sz]; }    };
private:
    vector<T> ve;
    SegmentTree<pair<T, int>> seg;
    vector<T> rv;
    vector<int> ri;
    bool build = false;
public:
    int n;
    ruimax(vector<T> &a) : ve(a), n(sz(a)) {        int index = -1;        T ma = MIN<T>();        rv.resize(n + 1);        ri.resize(n + 1);        rv[0] = -INF<T>;        ri[0] = -1;        rep(i, n) {            if (chma(ma, a[i])) { index = i; }            rv[i + 1] = ma;            ri[i + 1] = index;        }    }
    T operator()(int l, int r) {        if (!(l <= r && 0 <= l && r <= n)) {            deb(l, r, n);            assert(0);        }        if (l == 0) { return rv[r]; }        else {            if (!build)seg.build(ve), build = true;            return seg.query(l, r).first;        }    }
    T operator()(int r = inf) { return operator()(0, min(r, n)); }
    T operator[](int r) { return operator()(0, r); }
    T getv(int l, int r) { return operator()(l, r); }
    T getv(int r = inf) { return getv(0, min(r, n)); };
    int geti(int l, int r) {        assert(l <= r && 0 <= l && r <= n);        if (l == 0) { return ri[r]; }        else {            if (!build)seg.build(ve), build = true;            return seg.query(l, r).second;        }    }
    int geti(int r = inf) { return geti(0, min(r, n)); };
    auto begin() { return rv.begin(); }
    auto end() { return rv.end(); }
};
template<class T> struct ruimin {
    template<typename Monoid> struct SegmentTree {  /*pairで処理*/         int sz;vector<Monoid> seg;        const Monoid M1 = mp(MAX<T>(), -1);        Monoid f(Monoid a, Monoid b) { return min(a, b); }        void build(vector<T> &a) {            int n = sz(a);            sz = 1;            while (sz < n) sz <<= 1;            seg.assign(2 * sz, M1);            rep(i, n) { seg[i + sz] = mp(a[i], i); }            for (int k = sz - 1; k > 0; k--) { seg[k] = f(seg[k << 1], seg[(k << 1) | 1]); }        }        Monoid query(int a, int b) {            Monoid L = M1, R = M1;            for (a += sz, b += sz; a < b; a >>= 1, b >>= 1) {                if (a & 1) L = f(L, seg[a++]);                if (b & 1) R = f(seg[--b], R);            }            return f(L, R);        }        Monoid operator[](const int &k) const { return seg[k + sz]; }    };
private:
    vector<T> ve;
    SegmentTree<pair<T, int>> seg;
    vector<T> rv;
    vector<int> ri;
    bool build = false;
    int n;
public:
    ruimin(vector<T> &a) : ve(a), n(sz(a)) {        int index = -1;        T mi = MAX<T>();        rv.resize(n + 1);        ri.resize(n + 1);        rv[0] = INF<T>;        ri[0] = -1;        rep(i, n) {            if (chmi(mi, a[i])) { index = i; }            rv[i + 1] = mi;            ri[i + 1] = index;        }    }
    T operator()(int l, int r) {        assert(l <= r && 0 <= l && r <= n);        if (l == 0) { return rv[r]; }        else {            if (!build)seg.build(ve), build = true;            return seg.query(l, r).first;        }    }
    T operator()(int r = inf) { return operator()(0, min(r, n)); }
    T operator[](int r) { return operator()(0, r); }
    T getv(int l, int r) { return operator()(l, r); }
    T getv(int r = inf) { return getv(0, min(r, n)); };
    int geti(int l, int r) {        {            assert(l <= r && 0 <= l && r <= n);            if (l == 0) { return ri[r]; }            else {                if (!build)seg.build(ve), build = true;                return seg.query(l, r).second;            }        }        assert(l <= r && 0 <= l && r <= n);        if (l == 0) { return ri[r]; }        else {            if (!build)seg.build(ve), build = true;            return seg.query(l, r).second;        }    }
    int geti(int r = inf) { return geti(0, min(r, n)); };
    auto begin() { return rv.begin(); }
    auto end() { return rv.end(); }
};/*@formatter:off*/

vvi() ruib(vi &a) {    vvi(res, 61, sz(a) + 1);    rep(k, 61) { rep(i, sz(a)) { res[k][i + 1] = res[k][i] + ((a[i] >> k) & 1); }}    return res;}
vector<ruiC<int>> ruibc(vi &a) {vector<ruiC<int>> ret(61);    vvi(res, 61, sz(a));    rep(k, 61) {        rep(i, sz(a)) { res[k][i] = (a[i] >> k) & 1; }        ret[k] = ruic(res[k]);    }    return ret;}
vector<ll> ruiv(string &a) {    if (sz(a) == 0)return vi(1);    ll dec = ('0' <= a[0] && a[0] <= '9') ? '0' : 0;    vector<ll> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + a[i] - dec;    return ret;}
ruiC<ll> ruic(string &a) {    vector<ll> ret = ruiv(a);    return ruiC<ll>(ret);}
//kと同じものの数
template<class T, class U> vi ruiv(T &a, U k) {    vi ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + (a[i] == k);    return ret;}
template<class T, class U> ruiC<ll> ruic(T &a, U k) {    vi ret = ruiv(a, k);    return ruiC<ll>(ret);}
template<class T> struct ruiC2 {
    int H;
    vector<ruiC<T>> rui;
    ruiC2(vector<vector<T>> &ru) : rui(sz(ru)), H(sz(ru)) { for (int h = 0; h < H; h++) { rui[h] = ruic(ru[h]); }}
    //WについてHを返す
    vector<T> operator()(ll l, ll r) {        if (l > r) {            cerr << "ruic ";            deb(l, r);            assert(0);        }        vector<T> res(H);        for (int h = 0; h < H; h++)res[h] = rui[h](l, r);        return res;    }
    //HについてWを返す
    ruiC<T> &operator[](ll h) {        assert(h < H);        return rui[h];    }
    //    vector<T> operator()(int r) { return operator()(0, r); }
    /*ruiv[]をruic[]に変えた際意味が変わるのがまずいため()と統一*/
    /*単体iを返す 累積でないことに注意(seg木との統一でこうしている)*/
//    T operator[](ll i) { return rui[i + 1] - rui[i]; }
    /*0から順に追加される必要がある*/
//    T back() { return rui.back(); }
//    ll size() { return rui.size(); }
//    auto begin(){return rui.begin();}
//    auto end(){return rui.end();}
};

//a~zを0~25として
// rui(l,r)でvector(26文字について, l~rのcの個数)
// rui[h] ruic()を返す
ruiC2<ll> ruicou(str &a) {    str s = a;    replace(s);    vector<ruiC<ll>> res(26);    vvi(cou, 26, sz(s));    rep(i, sz(s)) { cou[s[i]][i] = 1; }    return ruiC2<ll>(cou);}
ruiC2<ll> ruicou(vi &a) {    int H = max(a) + 1;    vector<ruiC<ll>> res(H);    vvi(cou, H, sz(a));    rep(i, sz(a)) { cou[a[i]][i] = 1; }    return ruiC2<ll>(cou);}
template<class T, class U> ruiC<ll> ruicou(vector<T> &a, U b) {    vi cou(sz(a));    rep(i, sz(a)) { cou[i] = a[i] == b; }    return ruic(cou);}
//h query
template<class T> vector<T> imoh(vector<vector<T>> &v, int w) {    vector<T> ret(sz(v));    rep(h, sz(ret)) { ret[h] = v[h][w]; }    rep(i, sz(ret) - 1) { ret[i + 1] += ret[i]; }    return ret;}
template<class T> vector<T> ruih(vector<vector<T>> &v, int w) {    vector<T> ret(sz(v) + 1);    rep(h, sz(v)) { ret[h + 1] = v[h][w]; }    rep(i, sz(v)) { ret[i + 1] += ret[i]; }    return ret;}
template<class T> ruiC<T> ruihc(vector<vector<T>> &a, int w) {    vector<T> ret = ruih(a, w);    return ruiC<T>(ret);}

//xor
template<class T> struct ruixC {
    vector<T> rui;
    ruixC(vector<T> &ru) : rui(ru) {}
    T operator()(ll l, ll r) {        if (l > r) {            cerr << "ruiXc ";            deb(l, r);            assert(0);        }        return rui[r] ^ rui[l];    }
    T operator[](ll i) { return rui[i]; }
    T back() { return rui.back(); }
    ll size() { return rui.size(); }
};
template<class T> vector<T> ruix(vector<T> &a) {    vector<T> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] ^ a[i];    return ret;}
template<class T> ruixC<ll> ruixc(vector<T> &a) {    vi ret = ruix(a);    return ruixC<ll>(ret);}
//差分を返す(累積を取ると元に戻る)
//101なら
//1111を返す
//元の配列で[l, r)へのxorは
//[l]と[r]へのxorになる https://atcoder.jp/contests/abc155/tasks/abc155_f
vi ruix_diff(vi &A) {    int N = sz(A);    assert(N);    vi res(N + 1);    res[0] = A[0];    rep(i, 1, N) { res[i] = A[i - 1] ^ A[i]; }    res[N] = A[N - 1];    return res;}

template<class T> vector<T> ruim(vector<T> &a) {    vector<T> res(a.size() + 1, 1);    rep(i, a.size())res[i + 1] = res[i] * a[i];    return res;}
//漸化的に最小を1indexで持つ
template<class T> vector<T> ruimi(vector<T> &a) {ll n = sz(a);    vector<T> ret(n + 1);    rep(i, 1, n) {        ret[i] = a[i - 1];        chmi(ret[i + 1], ret[i]);    }    return ret;}
//template<class T> T *rrui(vector<T> &a) {
//右から左にかけての半開区間 (-1 n-1]
template<class T> struct rruiC {
    vector<T> rui;
    int n;
    rruiC(vector<T> &a) : n(sz(a)) {        rui.resize(n + 1);        rer(i, n - 1) { rui[i] = rui[i + 1] + a[i]; }    }
    /*[r l)*/
    T operator()(int r, int l) {        r++;        l++;        assert(l <= r && l >= 0 && r <= n);        return rui[l] - rui[r];    }
    T operator()(int l) { return operator()(n - 1, l); }
    T operator[](int i) { return operator()(i); }
};
template<class T> ostream &operator<<(ostream &os, rruiC<T> a) {    fora_f(v, a.rui){fora_f_init(v, a.rui);os << v << " "; }   return os;}

#define rrui rruic
template<class T> rruiC<T> rruic(vector<T> &a) { return rruiC<T>(a); }

//掛け算
template<class T> struct ruimulC {
    vector<T> rv;
    int n;
    ruimulC(vector<T> &a) : rv(a), n(sz(a)) {        rv.resize(n + 1);        rv[0] = 1;        rep(i, n) { rv[i + 1] = a[i] * rv[i]; }    }
    ruimulC() : n(0) {        rv.resize(n + 1);        rv[0] = 1;    }
    void operator+=(T v) {        rv.push_back(rv.back() * v);        n++;    }
    T operator()(int l, int r) {        assert(l <= r && 0 <= l && r <= n);        return rv[r] / rv[l];    }
    T operator()(int r = inf) { return operator()(0, min(r, n)); }
    T operator[](int r) { return operator()(0, r); }
    auto begin() { return rv.begin(); }
    auto end() { return rv.end(); }
};
template<class T> ruimulC<T> ruimul(vector<T> &a) { return ruimulC<T>(a); }
template<class T> ruimulC<T> ruimul() {    vector<T> a;    return ruimulC<T>(a);}
template<class T> T *rruim(vector<T> &a) {    ll len = a.size();    T *body = (T *) malloc((len + 1) * sizeof(T));    T *res = body + 1;    res[len - 1] = 1;    rer(i, len - 1)res[i - 1] = res[i] * a[i];    return res;}
template<class T, class U, class W> T lowerBound(ruiC <T> &a, U v, W banpei) { return lowerBound(a.rui, v, banpei); }
template<class T, class U, class W> T upperBound(ruiC <T> &a, U v, W banpei) { return upperBound(a.rui, v, banpei); }
template<class T, class U, class W> T rlowerBound(ruiC <T> &a, U v, W banpei) { return rlowerBound(a.rui, v, banpei); }
template<class T, class U, class W> T rupperBound(ruiC <T> &a, U v, W banpei) { return rupperBound(a.rui, v, banpei); }
#endif



//pow周りの仕様
//powiを使うと整数型
//powbを使うとbint型
//powを使うと powlに変換され long doubleが返る
#ifdef _DEBUG
//整数値の場合はpowiを使った方がいいというメッセージを出すための物
auto my_powl(ll a, ll k) { return powl(a, k); }
auto pow(ll a, ll k) {    static bool was = 1;    if (was) { message += "if integer use *powi* it's very fast\n"; }    was = 0;    return my_powl(a, k);}
//上のメッセージを出すための関数
auto pow(signed a, ll k) { return pow((ll) a, k); }
auto pow(signed a, signed k) { return pow((ll) a, (ll) k); }
#endif

//整数型のpow
int powi(int a, int k) {    if (a == 2)return 1ll << k;    int res = 1;    int x = a;    while (k) {        if (k & 1)res *= x;        x *= x;        k >>= 1;    }    return res;}
//define pow powlより上に動かすとバグる
bint pow(bint a, ll k) {    bint res = 1;    bint x = a;    while (k) {        if (k & 1)res *= x;        x *= x;        k >>= 1;    }    return res;}
bint pow(bint a, signed k) { return pow(a, (ll) k); }
bint powb(int a, int b) { return pow((bint) a, b); }
constexpr bool bget(ll m, ll keta) {
#ifdef _DEBUG
    assert(keta <= 62);//オーバーフロー 1^62までしか扱えない
#endif
    return (m >> keta) & 1;
}
//bget(n)次元
//                          NならN-1まで
vector<vi> bget2(vi &a, int keta_size) {    vvi(res, keta_size, sz(a));    rep(k, keta_size) { rep(i, sz(a)) { res[k][i] = bget(a[i], k); }}    return res;}
vi bget1(vi &a, int keta) {    vi res(sz(a));    rep(i, sz(a)) { res[i] = bget(a[i], keta); }    return res;}
ll bget(ll m, ll keta, ll sinsuu) {    m /= (ll) pow(sinsuu, keta);    return m % sinsuu;}
constexpr ll bit(ll n) {
#ifdef _DEBUG
    assert(n <= 62);//オーバーフロー 1^62までしか扱えない
#endif
    return (1LL << (n));
}
ll bit(ll n, ll sinsuu) { return (ll) pow(sinsuu, n); }
ll mask(ll n) { return (1ll << n) - 1; }
//aをbitに置きなおす
//{0, 2} -> 101
ll bit(const vi &a) {    int m = 0;    for (auto &&v:a) m |= bit(v);    return m;}

//{1, 1, 0} -> 011
//bitsetに置き換える感覚 i が立っていたら i bit目を立てる
ll bit_bool(vi &a) {    int m = 0;    rep(i, sz(a)) if (a[i])m |= bit(i);    return m;}
#define bcou __builtin_popcountll
//最下位ビット
ll lbit(ll n) { assert(n);return n & -n; }
ll lbiti(ll n) { assert(n);return log2(n & -n); }
//最上位ビット
ll hbit(ll n) {    assert(n);n |= (n >> 1);    n |= (n >> 2);    n |= (n >> 4);    n |= (n >> 8);    n |= (n >> 16);    n |= (n >> 32);    return n - (n >> 1);}
ll hbiti(ll n) { assert(n);return log2(hbit(n)); }
//ll hbitk(ll n) {    ll k = 0;    rer(i, 5) {        ll a = k + (1ll << i);        ll b = 1ll << a;        if (b <= n)k += 1ll << i;    }    return k;}
//初期化は0を渡す
ll nextComb(ll &mask, ll n, ll r) {    if (!mask)return mask = (1LL << r) - 1;    ll x = mask & -mask; /*最下位の1*/    ll y = mask + x; /*連続した下の1を繰り上がらせる*/    ll res = ((mask & ~y) / x >> 1) | y;    if (bget(res, n))return mask = 0; else return mask = res;}
//n桁以下でビットがr個立っているもののvectorを返す
vi bitCombList(ll n, ll r) {    vi res;    ll m = 0;    while (nextComb(m, n, r)) { res.push_back(m); }    return res;}
/*over*/#define forbit1_2(i, mas) for (int forbitj = !mas ? 0 : lbit(mas), forbitm = mas, i = !mas ? 0 :log2(forbitj); forbitm; forbitm = forbitm ^ forbitj, forbitj = !forbitm ? 1 : lbit(forbitm), i = log2(forbitj))
/*over*/#define forbit1_3(i, N, mas) for (int forbitj = !mas ? 0 : lbit(mas), forbitm = mas, i = !mas ? 0 :log2(forbitj); forbitm  && i < N; forbitm = forbitm ^ forbitj, forbitj = !forbitm ? 1 : lbit(forbitm), i = log2(forbitj))
//masの立ってるindexを見る
//              i, [N], mas
#define forbit1(...) over3(__VA_ARGS__, forbit1_3, forbit1_2)(__VA_ARGS__)
//masが立っていないindexを見る
//              i, N, mas
#define forbit0(i, N, mas) forbit1(i, mask(N) & (~(mas)))

//forsubをスニペットして使う
//Mの部分集合(0,M含む)を見る 3^sz(S)個ある
#define forsub_all(m, M) for (int m = M; m != -1; m = m == 0 ? -1 : (m - 1) & M)


//aにある物をtrueとする
vb bool_(vi a, int n) {    vb ret(max(max(a) + 1, n));    rep(i, sz(a))ret[a[i]] = true;    return ret;}

char itoal(ll i) { return 'a' + i; }
char itoaL(ll i) { return 'A' + i; }
ll altoi(char c) {    if ('A' <= c && c <= 'Z')return c - 'A';    return c - 'a';}
ll ctoi(char c) { return c - '0'; }
char itoc(ll i) { return i + '0'; }
ll vtoi(vi &v) {    ll res = 0;    if (sz(v) > 18) {        debugline("vtoi");        deb(sz(v));        ole();    }    rep(i, sz(v)) {        res *= 10;        res += v[i];    }    return res;}
vi itov(ll i) {    vi res;    while (i) {        res.push_back(i % 10);        i /= 10;    }    res = rev(res);    return res;}
vi stov(string &a) {    ll n = sz(a);    vi ret(n);    rep(i, n) { ret[i] = a[i] - '0'; }    return ret;}
//基準を満たさないものは0になる
vi stov(string &a, char one) {    ll n = sz(a);    vi ret(n);    rep(i, n)ret[i] = a[i] == one;    return ret;}
vector<vector<ll>> ctoi(vector<vector<char>> s, char c) {    ll n = sz(s), m = sz(s[0]);    vector<vector<ll>> res(n, vector<ll>(m));    rep(i, n)rep(j, m)res[i][j] = s[i][j] == c;    return res;}
//#define use_compress
//[i] := vを返す
//aは0~n-1で置き換えられる
vi compress(vi &a) {    vi b;    ll len = a.size();    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) { a[i] = lower_bound(all(b), a[i]) - b.begin(); }    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
#ifdef use_compress
//ind[i] := i番目に小さい数
//map[v] := vは何番目に小さいか
vi compress(vi &a, umapi &map) {    vi b;    ll len = a.size();    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) {        ll v = a[i];        a[i] = lower_bound(all(b), a[i]) - b.begin();        map[v] = a[i];    }    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vi &a, vi &r) {    vi b;    ll len = a.size();    fora_f(v, a){fora_f_init(v, a);b.push_back(v);}    fora_f(v, r){fora_f_init(v, r);b.push_back(v);}    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) a[i] = lower_bound(all(b), a[i]) - b.begin();    for (ll i = 0; i < sz(r); ++i) r[i] = lower_bound(all(b), r[i]) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vi &a, vi &r, vi &s) {    vi b;    ll len = a.size();    fora_f(v, a){fora_f_init(v, a);b.push_back(v);}    fora_f(v, r){fora_f_init(v, r);b.push_back(v);}    fora_f(v, s){fora_f_init(v, s);b.push_back(v); }   sort(b);    unique(b);    for (ll i = 0; i < len; ++i) a[i] = lower_bound(all(b), a[i]) - b.begin();    for (ll i = 0; i < sz(r); ++i) r[i] = lower_bound(all(b), r[i]) - b.begin();    for (ll i = 0; i < sz(s); ++i) r[i] = lower_bound(all(b), s[i]) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vector<vi> &a) {    vi b;    fora_f(vv, a){fora_f_init(vv, a);fora_f(v, vv){fora_f_init(v, vv);b.push_back(v);}}    sort(b);    unique(b);    fora_f(vv, a){fora_f_init(vv, a);fora_f(v, vv){fora_f_init(v, vv);v = lower_bound(all(b), v) - b.begin(); }}   ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vector<vector<vi >> &a) {    vi b;    fora_f(vvv, a){fora_f_init(vvv, a);fora_f(vv, vvv){fora_f_init(vv, vvv);fora_f(v, vv){fora_f_init(v, vv);b.push_back(v);}}}    sort(b);    unique(b);   fora_f(vvv, a){fora_f_init(vvv, a);fora_f(vv, vvv){fora_f_init(vv, vvv);fora_f(v, vv){fora_f_init(v, vv);v = lower_bound(all(b), v) - b.begin();}}}    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
void compress(ll a[], ll len) {    vi b;    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) { a[i] = lower_bound(all(b), a[i]) - b.begin(); }}
#endif
//要素が見つからなかったときに困る
#define binarySearch(a, v) (binary_search(all(a),v))
#define lowerIndex(a, v) (lower_bound(all(a),v)-a.begin())
#define upperIndex(a, v) (upper_bound(all(a),v)-a.begin())
#define rlowerIndex(a, v) (upper_bound(all(a),v)-a.begin()-1)
#define rupperIndex(a, v) (lower_bound(all(a),v)-a.begin()-1)

template<class T, class U, class W> T lowerBound(vector<T> &a, U v, W banpei) {    auto it = lower_bound(a.begin(), a.end(), v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T upperBound(vector<T> &a, U v, W banpei) {    auto it = upper_bound(a.begin(), a.end(), v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T rlowerBound(vector<T> &a, U v, W banpei) {    auto it = upper_bound(a.begin(), a.end(), v);    if (it == a.begin())return banpei; else { return *(--it); }}
template<class T, class U, class W> T rupperBound(vector<T> &a, U v, W banpei) {    auto it = lower_bound(a.begin(), a.end(), v);    if (it == a.begin())return banpei; else { return *(--it); }}

template<class T, class U, class W> T lowerBound(set<T> &a, U v, W banpei) {    auto it = a.lower_bound(v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T upperBound(set<T> &a, U v, W banpei) {    auto it = a.upper_bound(v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T rlowerBound(set<T> &a, U v, W banpei) {    auto it = a.upper_bound(v);    if (it == a.begin())return banpei; else { return *(--it); }}
template<class T, class U, class W> T rupperBound(set<T> &a, U v, W banpei) {auto it = a.lower_bound(v);if (it == a.begin())return banpei; else { return *(--it); }}

template<class T, class U, class W> T lowerBound(mset<T> &a, U v, W banpei) {    auto it = a.lower_bound(v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T upperBound(mset<T> &a, U v, W banpei) {    auto it = a.upper_bound(v);    if (it == a.end())return banpei; else return *it;}
template<class T, class U, class W> T rlowerBound(mset<T> &a, U v, W banpei) {    auto it = a.upper_bound(v);    if (it == a.begin())return banpei; else { return *(--it); }}
template<class T, class U, class W> T rupperBound(mset<T> &a, U v, W banpei) {auto it = a.lower_bound(v);if (it == a.begin())return banpei; else { return *(--it); }}
#define next2(a) next(next(a))
#define prev2(a) prev(prev(a))

//狭義の単調増加列 長さを返す
template<class T> int lis(vector<T> &a) {    int n = sz(a);    vi tail(n + 1, MAX<T>());    rep(i, n) {        int id = lowerIndex(tail, a[i]);/**/        tail[id] = a[i];    }    return lowerIndex(tail, MAX<T>());}
template<class T> int lis_eq(vector<T> &a) {    int n = sz(a);    vi tail(n + 1, MAX<T>());    rep(i, n) {        int id = upperIndex(tail, a[i]);/**/        tail[id] = a[i];    }    return lowerIndex(tail, MAX<T>());}

//iteratorを返す
//valueが1以上の物を返す 0は見つけ次第削除
//vを減らす場合 (*it).se--でいい
template<class T, class U, class V> auto lower_map(map<T, U> &m, V k) {    auto ret = m.lower_bound(k);    while (ret != m.end() && (*ret).second == 0) {        ret = m.erase(ret);    }    return ret;}
template<class T, class U, class V> auto upper_map(map<T, U> &m, V k) {    auto ret = m.upper_bound(k);    while (ret != m.end() && (*ret).second == 0) {        ret = m.erase(ret);    }    return ret;}
//存在しなければエラー
template<class T, class U, class V> auto rlower_map(map<T, U> &m, V k) {    auto ret = upper_map(m, k);    assert(ret != m.begin());    ret--;    while (1) {        if ((*ret).second != 0)break;        assert(ret != m.begin());        auto next = ret;        --next;        m.erase(ret);        ret = next;    }    return ret;}
template<class T, class U, class V> auto rupper_map(map<T, U> &m, V k) {    auto ret = lower_map(m, k);    assert(ret != m.begin());    ret--;    while (1) {        if ((*ret).second != 0)break;        assert(ret != m.begin());        auto next = ret;        --next;        m.erase(ret);        ret = next;    }    return ret;}

template<class... T> void fin(T... s) {out(s...); exit(0); }

//便利 数学 math
//sub ⊂ top
bool subset(int sub, int top) {return (sub & top) == sub;}
//-180 ~ 180 degree
double atand(double h, double w) {return atan2(h, w) / PI * 180;}
//% -mの場合、最小の正の数を返す
ll mod(ll a, ll m) {if (m < 0) m *= -1;return (a % m + m) % m;}
ll pow(ll a) { return a * a; };
ll fact(ll v) { return v <= 1 ? 1 : v * fact(v - 1); }
dou factd(int v){static vd fact(2,1);    if(sz(fact)<=v){        rep(i,sz(fact),v+1){            fact.push_back(fact.back()*i);        }    }    return fact[v];}

ll comi(ll n, ll r) {    assert(n < 100);    static vvi(pas, 100, 100);    if (pas[0][0])return pas[n][r];    pas[0][0] = 1;    rep(i, 1, 100) {        pas[i][0] = 1;        rep(j, 1, i + 1)pas[i][j] = pas[i - 1][j - 1] + pas[i - 1][j];    }    return pas[n][r];}
//二項係数の偶奇を返す
int com_mod2(int n,int r){return n == ( r | (n - r) );}
double comd2(ll n, ll r) {    static vvd(comb, 2020, 2020);    if (comb[0][0] == 0) {        comb[0][0] = 1;        rep(i, 2000) {            comb[i + 1][0] = 1;            rep(j, 1, i + 2) { comb[i + 1][j] = comb[i][j] + comb[i][j - 1]; }        }    }    return comb[n][r];}
double comd(int n, int r) {    if (r < 0 || r > n) return 0;    if (n < 2020)return comd2(n, r);    static vd fact(2, 1);    if (sz(fact) <= n) { rep(i, sz(fact), n + 1) { fact.push_back(fact.back() * i); }}    return fact[n] / fact[n - r] / fact[r];}

#define gcd my_gcd
#define lcm my_lcm
ll gcd(ll a, ll b) {while (b) a %= b, swap(a, b);return abs(a);}
ll gcd(vi b) {ll res = b[0];rep(i, 1, sz(b))res = gcd(b[i], res);return res;}
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
ll lcm(vi a) {ll res = a[0];rep(i, 1, sz(a))res = lcm(a[i], res);return res;}
ll ceil(ll a, ll b) {if (b == 0) {debugline("ceil");deb(a, b);ole();return -1;} else if (a < 0) { return 0; } else { return (a + b - 1) / b; }}

ll sig0(int t) { return t <= 0 ? 0 : ((1 + t) * t) >> 1; }
bint sig0(bint t) {return t <= 0 ? 0 : ((1 + t) * t) >> 1; }
//ll sig(ll s, ll t) { return ((s + t) * (t - s + 1)) >> 1; }
ll sig(ll s, ll t) {if (s > t)swap(s, t);return sig0(t - s) + s * (t - s + 1);}

#define tousa_i tosa_i
#define lower_tousa_i lower_tosa_i
#define upper_tousa upper_tosa
#define upper_tousa_i upper_tosa_i
ll tosa_i(ll st, ll ad, ll v) {    assert(((v - st) % ad) == 0);    return (v - st) / ad;}
ll tosa_s(ll st, ll ad, ll len) {    return st * len + sig0(len - 1) * ad;}

// ax + r (x は非負整数) で表せる整数のうち、v 以上となる最小の整数
ll lower_tosa(ll st, ll ad, ll v) {    if (st >= v) return st;    return (v - st + ad - 1) / ad * ad + st;}
//第何項か
ll lower_tosa_i(ll st, ll ad, ll v) {    if (st >= v) return 0;    return (v - st + ad - 1) / ad;}
ll upper_tosa(ll st, ll ad, ll v) { return lower_tosa(st, ad, v + 1); }
ll upper_tosa_i(ll st, ll ad, ll v) { return lower_tosa_i(st, ad, v + 1); }

//b * res <= aを満たす [l, r)を返す div
P drange_ika(int a, int b) {    P null_p = mp(linf, linf);    if (b == 0) {        if (a >= 0) {            return mp(-linf, linf + 1)/*全て*/;        } else {            return null_p/*無い*/;        }    } else {        if (a >= 0) {            if (b > 0) {                return mp(-linf, a / b + 1);            } else {                return mp(-(a / -b), linf + 1);            }        } else {            if (b > 0) {                return mp(-linf, -ceil(-a, b) + 1);            } else {                return mp(ceil(-a, -b), linf + 1);            }        }    }}

//v * v >= aとなる最小のvを返す
ll sqrt(ll a) {    if (a < 0) {        debugline("sqrt");        deb(a);        ole();    }    ll res = (ll) std::sqrt(a);    while (res * res < a)++res;    return res;}
double log(double e, double x) { return log(x) / log(e); }


/*@formatter:off*/

//機能拡張
#define dtie(a, b) int a, b; tie(a, b)
template<class T, class U> string to_string(T a, U b) {    string res = "";    res += a;    res += b;    return res;}
template<class T, class U, class V> string to_string(T a, U b, V c) {    string res = "";    res += a;    res += b;    res += c;    return res;}
template<class T, class U, class V, class W> string to_string(T a, U b, V c, W d) {    string res = "";    res += a;    res += b;    res += c;    res += d;    return res;}
template<class T, class U, class V, class W, class X> string to_string(T a, U b, V c, W d, X e) {    string res = "";    res += a;    res += b;    res += c;    res += d;    res += e;    return res;}

template<class T> vector<T> sub(const vector<T> &A, int l, int r) {    assert(0 <= l && l <= r && r <= sz(A));    vector<T> ret(r - l);    std::copy(A.begin() + l, A.begin() + r, ret.begin());    return ret;}
template<class T> vector<T> sub(const vector<T> &A, int r) { return sub(A, 0, r); }
template<class T> vector<T> subn(const vector<T> &A, int l, int len) { return sub(A, l, l + len); }
string sub(string &A, int l, int r) {    assert(0 <= l && l <= r && r <= sz(A));    return A.substr(l, r - l);}


constexpr int bsetlen = k5 * 2;
//constexpr int bsetlen = 5050;
#define bset bitset<bsetlen>
bool operator<(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] < b[i])return true;if (a[i] > b[i])return false;}return false;}
bool operator>(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] > b[i])return true;if (a[i] < b[i])return false;}return false;}
bool operator<=(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] < b[i])return true;if (a[i] > b[i])return false;}return true;}
bool operator>=(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] > b[i])return true;if (a[i] < b[i])return false;}return true;}
string operator~(string &a) {string res = a;for (auto &&c:res) {if (c == '0')c = '1';else if (c == '1')c = '0';else {cerr << "cant ~" << a << "must bit" << endl;exit(0);}}return res;}
ostream &operator<<(ostream &os, bset& a) {    bitset<10> b;    vi list;    rep(i,bsetlen){        if(a[i])list.push_back(i),b[i]=1;    }    os<<b<<", "<<list;    return os;}
int hbiti(bset&a){rer(i,bsetlen){if(a[i])return i;}return -1;}
#define hk(a, b, c) (a <= b && b < c)

//O(N/64)
bset nap(bset &a, int v) {bset r = a | a << v;return r;}
bset nap(bset &a, bset &v) {bset r = a;rep(i, bsetlen) {if (v[i])r |= a << i;}return r;}

template<class T> int count(set<T> &S, T l, T r) {    assert(l < r);    auto it = S.lower_bound(l);    return it != S.end() && (*it) < r;}

template<class T> void seth(vector<vector<T>> &S, int w, vector<T> &v) {assert(sz(S) == sz(v));assert(w < sz(S[0]));rep(h, sz(S)) { S[h][w] = v[h]; }}

template<class T, class U> void operator+=(pair<T,U> &a, pair<T,U> & b) {a.fi+=b.fi;a.se+=b.se;}
template<class T, class U> pair<T,U> operator+(pair<T,U> &a, pair<T,U> & b) {return pair<T,U>(a.fi+b.fi,a.se+b.se);}
template<typename CharT, typename Traits, typename Alloc> basic_string<CharT, Traits, Alloc> operator+(const basic_string<CharT, Traits, Alloc> &lhs, const int rv) {
#ifdef _DEBUG
    static bool was = false;if (!was)message += "str += 65  is  'A'  not \"65\" ";was = true;
#endif
    return lhs + (char) rv;
}
template<typename CharT, typename Traits, typename Alloc> void operator+=(basic_string<CharT, Traits, Alloc> &lhs, const int rv) {    lhs = lhs + rv;}
template<typename CharT, typename Traits, typename Alloc> basic_string<CharT, Traits, Alloc> operator+(const basic_string<CharT, Traits, Alloc> &lhs, const signed rv) {    const int rv2 = rv;    return lhs + rv2;}
template<typename CharT, typename Traits, typename Alloc> void operator+=(basic_string<CharT, Traits, Alloc> &lhs, const signed rv) {const int v = rv; lhs += v; }
template<typename CharT, typename Traits, typename Alloc> void operator*=(basic_string<CharT, Traits, Alloc> &s, int num) {    auto bek = s;    s = "";    for (; num; num >>= 1) {        if (num & 1) { s += bek; }        bek += bek;    }}
template<class T, class U> void operator+=(queue<T> &a, U v) { a.push(v); }template<class T, class U> void operator+=(deque<T> &a, U v) { a.push_back(v); }template<class T> priority_queue<T, vector<T>, greater<T> > &operator+=(priority_queue<T, vector<T>, greater<T> > &a, vector<T> &v) {    fora_f(d, v){fora_f_init(d, v);a.push(d);}    return a;}template<class T, class U> priority_queue<T, vector<T>, greater<T> > &operator+=(priority_queue<T, vector<T>, greater<T> > &a, U v) {    a.push(v);    return a;}template<class T, class U> priority_queue<T> &operator+=(priority_queue<T> &a, U v) {    a.push(v);    return a;}template<class T> set<T> &operator+=(set<T> &a, vector<T> v) {    fora_f(d, v){fora_f_init(d, v);a.insert(d);}    return a;}template<class T, class U> auto operator+=(set<T> &a, U v) { return a.insert(v); }template<class T, class U> auto operator-=(set<T> &a, U v) { return a.erase(v); }template<class T, class U> auto operator+=(mset<T> &a, U v) { return a.insert(v); }template<class T, class U> set<T, greater<T>> &operator+=(set<T, greater<T>> &a, U v) {    a.insert(v);    return a;}template<class T, class U> vector<T> &operator+=(vector<T> &a, U v) {    a.push_back(v);    return a;}template<class T, class U> vector<T> operator+(U v,const vector<T> &a) {    vector<T> ret = a;    ret.insert(ret.begin(), v);    return ret;}template<class T> vector<T> operator+(const vector<T>& a, const vector<T>& b) {    vector<T> ret;    ret = a;    fora_f(v, b){fora_f_init(v, b);ret += v; }   return ret;}template<class T> vector<T> &operator+=(vector<T> &a,const vector<T> &b) {    rep(i, sz(b)) {/*こうしないとa+=aで両辺が増え続けてバグる*/        a.push_back(b[i]); }    return a;}template<class T, class U> map<T, U> &operator+=(map<T, U> &a, map<T, U> &b) {    for(auto&& bv : b) { a[bv.first] += bv.second; }    return a;}template<class T, class U> vector<T> operator+(const vector<T> &a, const U& v) {    vector<T> ret = a;    ret += v;    return ret;}
template<class T> vector<T> operator%(vector<T>& a, int v){    vi ret(sz(a));    rep(i,sz(a)){        ret[i] = a[i] % v;    }    return ret;}
template<class T> vector<T> operator%=(vector<T>& a, int v){    rep(i,sz(a)){        a[i] %= v;    }    return a;}
vi operator&(vi& a, vi& b){    assert(sz(a)==sz(b));    vi ret(sz(a));    rep(i,sz(a)){        ret[i] = min(a[i],b[i]);    }    return ret;}
template<class T> void operator+=(mset<T> &a, vector<T>& v) { for(auto&& u : v)a.insert(u); }
template<class T> void operator+=(set<T> &a, vector<T>& v) { for(auto&& u : v)a.insert(u); }
template<class T> void operator+=(vector<T> &a, set<T>& v) { for(auto&& u : v)a.emplace_back(u); }
template<class T> void operator+=(vector<T> &a, mset<T>& v) { for(auto&& u : v)a.emplace_back(u); }
template<class T> vector<T> &operator-=(vector<T> &a, vector <T> &b) {    if (sz(a) != sz(b)) {        debugline("vector<T> operator-=");        deb(a);        deb(b);        exit(0);    }    rep(i, sz(a))a[i] -= b[i];    return a;}
template<class T> vector<T> operator-(vector<T> &a, vector<T> &b) {    if (sz(a) != sz(b)) {        debugline("vector<T> operator-");        deb(a);        deb(b);        ole();    }    vector<T> res(sz(a));    rep(i, sz(a))res[i] = a[i] - b[i];    return res;}
//template<class T, class U> void operator*=(vector<T> &a, U b) {    vector<T> ta = a;    rep(b-1){        a+=ta;    }}
template<typename T> void erase(vector<T> &v, unsigned ll i) { v.erase(v.begin() + i); }
template<typename T> void erase(vector<T> &v, unsigned ll s, unsigned ll e) { v.erase(v.begin() + s, v.begin() + e); }
template<typename T> void pop_front(vector<T> &v) { erase(v, 0); }
template<typename T> void entry(vector<T> &v, unsigned ll s, unsigned ll e) { erase(v, e, sz(v));erase(v,0,s);}

template<class T, class U> void erase(map<T, U> &m, ll okl, ll ngr) { m.erase(m.lower_bound(okl), m.lower_bound(ngr)); }
template<class T> void erase(set<T> &m, ll okl, ll ngr) { m.erase(m.lower_bound(okl), m.lower_bound(ngr)); }
template<typename T> void erasen(vector<T> &v, unsigned ll s, unsigned ll n) { v.erase(v.begin() + s, v.begin() + s + n); }
template<typename T, typename U> void insert(vector<T> &v, unsigned ll i, U t) { v.insert(v.begin() + i, t); }
template<typename T, typename U> void push_front(vector<T> &v, U t) { v.insert(v.begin(), t); }
template<typename T, typename U> void insert(vector<T> &v, unsigned ll i, vector<T> list) { for (auto &&va:list)v.insert(v.begin() + i++, va); }
template<typename T> void insert(set<T> &v, vector<T> list) { for (auto &&va :list)v.insert(va); }
template<class T> T poll(set<T>& S){T ret = *S.begin();S.erase(S.begin());return ret;}
template<class T> T poll(mset<T>& S){T ret = *S.begin();S.erase(S.begin());return ret;}
template<class T> T poll_back(set<T>& S){T ret = *S.rbegin();S.erase(S.rbegin());return ret;}
template<class T> T poll_back(mset<T>& S){T ret = *S.rbegin();S.erase(S.rbegin());return ret;}
template<class T> T peek(set<T>& S){T ret = *S.begin();return ret;}
template<class T> T peek(mset<T>& S){T ret = *S.begin();return ret;}
template<class T> T peek_back(set<T>& S){T ret = *S.rbegin();return ret;}
template<class T> T peek_back(mset<T>& S){T ret = *S.rbegin();return ret;}
vector<string> split(const string a, const char deli) {    string b = a + deli;    ll l = 0, r = 0, n = b.size();    vector<string> res;    rep(i, n) {        if (b[i] == deli) {            r = i;            if (l < r)res.push_back(b.substr(l, r - l));            l = i + 1;        }    }    return res;}
vector<string> split(const string a, const string deli) {    vector<string> res;    ll kn = sz(deli);    std::string::size_type Pos(a.find(deli));    ll l = 0;    while (Pos != std::string::npos) {        if (Pos - l)res.push_back(a.substr(l, Pos - l));        l = Pos + kn;        Pos = a.find(deli, Pos + kn);    }    if (sz(a) - l)res.push_back(a.substr(l, sz(a) - l));    return res;}
ll stoi(string& s){return stol(s);}
#define assert_yn(yn_v, v); assert(yn_v == 0 || yn_v == v);yn_v = v;
//不完全な対策、現状はautohotkeyで対応
int yn_v = 0;
void yn(bool a) { assert_yn(yn_v, 1);if (a)cout << "yes" << endl; else cout << "no" << endl; }
void fyn(bool a) {    assert_yn(yn_v, 1);yn(a);    exit(0);}

void Yn(bool a) { assert_yn(yn_v, 2);if (a)cout << "Yes" << endl; else cout << "No" << endl; }
void fYn(bool a) {    assert_yn(yn_v, 2);Yn(a);    exit(0);}

void YN(bool a) { assert_yn(yn_v, 3);if (a)cout << "YES" << endl; else cout << "NO" << endl; }
void fYN(bool a) {    assert_yn(yn_v, 3);YN(a);    exit(0);}
int ab_v = 0;
void fAb(bool a) { assert_yn(ab_v, 1);if(a)cout<<"Alice"<<endl;else cout<<"Bob";}
void fAB(bool a) { assert_yn(yn_v, 2);if(a)cout<<"ALICE"<<endl;else cout<<"BOB";}
int pos_v = 0;
void Possible(bool a) {    assert_yn(pos_v, 1);if (a)cout << "Possible" << endl; else cout << "Impossible" << endl;    exit(0);}
void POSSIBLE(bool a) {    assert_yn(pos_v, 2);if (a)cout << "POSSIBLE" << endl; else cout << "IMPOSSIBLE" << endl;    exit(0);}
void fPossible(bool a) {    assert_yn(pos_v, 1)Possible(a);exit(0);}
void fPOSSIBLE(bool a) {    assert_yn(pos_v, 2)POSSIBLE(a);exit(0);}
template<typename T> class fixed_point        : T {public:    explicit constexpr fixed_point(T &&t) noexcept: T(std::forward<T>(t)) {}    template<typename... Args> constexpr decltype(auto) operator()(Args &&... args) const { return T::operator()(*this, std::forward<Args>(args)...); }};template<typename T> static inline constexpr decltype(auto) fix(T &&t) noexcept { return fixed_point<T>{std::forward<T>(t)}; }


//未分類
//0,2,1 1番目と2番目の次元を入れ替える
template<class T> auto irekae(vector<vector<vector<T> > > &A, int x, int y, int z) {
#define irekae_resize_loop(a, b, c) resize(res,a,b,c);rep(i,a)rep(j,b)rep(k,c)
    vector<vector<vector<T> > > res;    if (x == 0 && y == 1 && z == 2) {        res = A;    } else if (x == 0 && y == 2 && z == 1) {        irekae_resize_loop(sz(A), sz(A[0][0]), sz(A[0])) {                    res[i][j][k] = A[i][k][j];                }    } else if (x == 1 && y == 0 && z == 2) {        irekae_resize_loop(sz(A[0]), sz(A), sz(A[0][0])) {                    res[i][j][k] = A[j][i][k];                }    } else if (x == 1 && y == 2 && z == 0) {        irekae_resize_loop(sz(A[0]), sz(A[0][0]), sz(A)) {                    res[i][j][k] = A[k][i][j];                }    } else if (x == 2 && y == 0 && z == 1) {        irekae_resize_loop(sz(A[0][0]), sz(A), sz(A[0])) {                    res[i][j][k] = A[j][k][i];                }    } else if (x == 2 && y == 1 && z == 0) {        irekae_resize_loop(sz(A[0][0]), sz(A[0]), sz(A)) {                    res[i][j][k] = A[k][j][i];                }    }    return res;
#undef irekae_resize_loop
}
template<class T> auto irekae(vector<vector<T>>&A,int i=1,int j=0){    vvt(res,sz(A[0]),sz(A));    rep(i,sz(A)){        rep(j,sz(A[0])){            res[j][i]=A[i][j];        }    }    return res;}
//tou分割する
template<typename T> vector<vector<T>> cut(vector<T> &a, int tou = 2) {    int N = sz(a);    vector<vector<T>> res(tou);    int hab = N / tou;    vi lens(tou, hab);    rep(i, N % tou) { lens[tou - 1 - i]++; }    int l = 0;    rep(i, tou) {        int len = lens[i];        int r = l + len;        res[i].resize(len);        std::copy(a.begin() + l, a.begin() + r, res[i].begin());        l = r;    }    return res;}
//長さn毎に分割する
template<typename T> vector<vector<T>> cutn(vector<T> &a, int len) {    int N = sz(a);    vector<vector<T>> res(ceil(N, len));    vi lens(N / len, len);    if (N % len)lens.push_back(N % len);    int l = 0;    rep(i, sz(lens)) {        int len = lens[i];        int r = l + len;        res[i].resize(len);        std::copy(a.begin() + l, a.begin() + r, res[i].begin());        l = r;    }    return res;}
//縦を返す
vi& geth(vvi()& a, int w){    static vi ret; ret.resize(sz(a));   rep(i,sz(a)){        ret[i] = a[i][w];    }    return ret;}

//@起動時
struct initon {
    initon() {
        cin.tie(0);
        ios::sync_with_stdio(false);
        cout.setf(ios::fixed);
        cout.precision(16);
        srand((unsigned) clock() + (unsigned) time(NULL));
    };
} initonv;
//#define pre prev
//#define nex next

//gra mll pr
//上下左右
const string udlr = "udlr";
string UDLR = "UDLR";//x4と連動 UDLR.find('U') := x4[0]
vc atoz = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x','y', 'z'};
vc AtoZ = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z'};
//右、上が正
constexpr ll h4[] = {1, -1, 0, 0};
constexpr ll w4[] = {0, 0, -1, 1};
constexpr ll h8[] = {0, 1, 0, -1, -1, 1, 1, -1};
constexpr ll w8[] = {1, 0, -1, 0, 1, -1, 1, -1};
int mei_inc(int h, int w, int H, int W, int i) {while (++i < 4) { if (inside(h + h4[i], w + w4[i], H, W))return i; }return i;}
#define mei(nh, nw, h, w) for (int i = mei_inc(h, w, H, W, -1), nh = i<4? h + h4[i] : 0, nw = i<4? w + w4[i] : 0; i < 4; i=mei_inc(h,w,H,W,i), nh = h+h4[i], nw = w+w4[i])
int mei_inc8(int h, int w, int H, int W, int i) {    while (++i < 8) {        if (inside(h + h8[i], w + w8[i], H, W))return i;    }    return i;}
#define mei8(nh, nw, h, w) for (int i = mei_inc8(h, w, H, W, -1), nh = i<8? h + h8[i] : 0, nw = i<8? w + w8[i] : 0; i < 8; i=mei_inc8(h,w,H,W,i), nh = h+h8[i], nw = w+w8[i])
int mei_incv(int h, int w, int H, int W, int i, vp &p) {    while (++i < sz(p)) { if (inside(h + p[i].fi, w + p[i].se, H, W))return i; }    return i;}
#define meiv(nh, nw, h, w, p) for (int i = mei_incv(h, w, H, W, -1, p), nh = i<sz(p)? h + p[i].fi : 0, nw = i<sz(p)? w + p[i].se : 0; i < sz(p); i=mei_incv(h,w,H,W,i,p), nh = h+p[i].fi, nw = w+p[i].se)

//H*Wのグリッドを斜めに分割する
//右上
vector<vp> naname_list_ne(int H, int W) {    vector<vp> res(H + W - 1);    rep(sh, H) {        int sw = 0;        res[sh] += mp(sh, sw);        int nh = sh;        int nw = sw;        while (1) {            nh--;            nw++;            if (0 <= nh && nw < W) {                res[sh] += mp(nh, nw);            }else{                break;            }        }    }    rep(sw, 1, W) {        int sh = H - 1;        res[H + sw - 1] += mp(sh, sw);        int nh = sh;        int nw = sw;        while (1) {            nh--;            nw++;            if (0 <= nh && nw < W) {                res[H + sw-1] += mp(nh, nw);            }else{                break;            }        }    }    return res;}
//右下
vector<vp> naname_list_se(int H, int W) {    vector<vp> res(H + W - 1);    rep(sh, H) {        int sw = 0;        res[sh] += mp(sh, sw);        int nh = sh;        int nw = sw;        while (1) {            nh++;            nw++;            if (0 <= nh && nh< H && nw < W) { res[sh] += mp(nh, nw); } else { break; }        }    }    rep(sw, 1, W) {        int sh = 0;        res[H + sw - 1] += mp(sh, sw);        int nh = sh;        int nw = sw;        while (1) {            nh++;            nw++;            if (0 <= nh && nh < H && nw < W) { res[H + sw - 1] += mp(nh, nw); } else { break; }        }    }    return res;}

//グラフ内で #undef getid
//#define getidとしているため、ここを書き直したらgraphも書き直す
#define getid_2(h, w) ((h) * (W) + (w))
#define getid_1(p) ((p).first * W + (p).second)
#define getid(...) over2(__VA_ARGS__, getid_2, getid_1) (__VA_ARGS__)
#define getp(id) mp(id / W, id % W)
//#define set_shuffle() std::random_device seed_gen;std::mt19937 engine(seed_gen())
//#define shuffle(a) std::shuffle((a).begin(), (a).end(), engine);
//1980 開始からtime ms経っていたらtrue
vb bit_bool(int v, int len) {    assert(bit(len) > v);    vb ret(len);    rep(i, len) { ret[i] = bget(v, i); }    return ret;}
vi &range(int l, int r) {static vi ret;ret.resize(r - l);rep(v, l, r) {ret[v - l] = v;}return ret;}
vi &range(int r) {return range(0, r);}
vi tov(vb& a){    vi ret;    rep(i,sz(a)){        if(a[i])ret.push_back(i);    }    return ret;}
bool kaibun(const str& S){return S==rev(S);}
/*@formatter:on*/
template<class T> vector<T> repeat(const vector<T> &A, int kaisu) {
    vector<T> ret;
    while (kaisu--) {
        ret += A;
    }
    return ret;
}
/*@formatter:off*/
#define rge range
#define upd update
//S[{s, t, d}]
#define strs slice_str
struct slice_str {
    string S;
    slice_str() {}
    slice_str(const string &S) : S(S) {}
    slice_str(int len, char c) : S(len, c) {}
    auto size(){return S.size();}
    char& operator[](int p) {        return S[p]; }
    string operator[](initializer_list<int> p) {        if (sz(p) == 1) {            return S.substr(0, *(p.begin()));        } else if (sz(p) == 2) {            int l = *(p.begin());            int r = *(next(p.begin()));            return S.substr(l, r - l);        } else {            auto it = p.begin();            int s = *(it++);            int t = *(it++);            int d = *(it);            if (d == -1) {                int s_ = sz(S) - s - 1;                int t_ = sz(S) - t - 1;                return rev(S).substr(s_, t_ - s_);            } else if (d < 0) {                t = max(-1ll, t);                string ret;                while (s > t) {                    ret += S[s];                    s += d;                }                return ret;            } else {                t = min(sz(S), t);                string ret;                while (s < t) {                    ret += S[s];                    s += d;                }                return ret;            }        }    }
    operator string &() {        return S;    }
    template<class T> void operator+=(const T &a) { S += a; }
};
ostream &operator<<(ostream &os, const slice_str &a) {    os << a.S;    return os;}
istream &operator>>(istream &iss, const slice_str &a) {    iss >> a.S;    return iss;}
template<class T> bool can(const T &v, int i) { return 0 <= i && i < sz(v); }
//todo c++17
//template<class... T> auto sum(const T &... v) {return (v + ...);}
//† ←template終了
/*@formatter:on*/
//vectorで取れる要素数
//bool=> 1e9 * 8.32
//int => 1e8 * 2.6
//ll  => 1e8 * 1.3
//3次元以上取るとメモリがヤバい
//static配列を使う
#define VEC vector
vvc (ba);
ll N, M, H, W;
vi A, B, C;


void solve(){
    int Q;
    in(N, Q);
    dna3(S, T, X, N);
    rep(i, N)S[i] -= X[i], T[i] -= X[i];
    sortt(X, S, T);
    deb(S);
    deb(T);
    seti st;
    dna(D, Q);
    mapi ind;
    rep(i, Q){
        ind[D[i]] = i;
        st += D[i];
    }
    st += linf;
    deb(ind);
    deb(st);
    vi res(Q);
    rep( i, N){
        dtie(s, t ) = mp(S[i], T[i]);
        auto it = st. lower_bound(s);
        while((*it) < t){
            res[ind[*it]] = X[i];
            it = st.erase(it);
        }
    };
    auto it = st.beg();
    while(it != st.end()){
        if(*it == linf)break;
        res[ind[*it]]=-1;
        it++;
    }
    outl(res);

}


auto my(ll n, vi &a) {
    return 0;
}

auto sister(ll n, vi &a) {
    ll ret = 0;
    return ret;
}

signed main() {
    solve();

#define arg n,a
#ifdef _DEBUG

    bool bad = 0;
    for (ll i = 0, ok = 1; i < k5 && ok; ++i) {
        ll n = rand(1, 8);
        vi a = ranv(n, 1, 10);
        auto myres = my(arg);
        auto res = sister(arg);
        ok = myres == res;
        if (!ok) {
            out(arg);
            cerr << "AC : " << res << endl;
            cerr << "MY  : " << myres << endl;
            bad = 1;


            break;
        }
    }
    if (!bad) {
//        cout << "完璧 : solveを書き直そう" << endl;
//        cout << "     : そして、solve()を呼び出すのだ" << endl;
//        cout << "     : cin>>n; na(a,n);も忘れるな" << endl;
    }
    if (was_deb && sz(res_mes)) {
        cerr << "result = " << endl << res_mes << endl;
    }
    if (sz(message)) {
        cerr << "****************************" << endl;
        cerr << "Note." << endl;
        cerr << message << endl;
        cerr << "****************************" << endl;
    }
#endif
    return 0;
};

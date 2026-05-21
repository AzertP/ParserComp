//#undef _DEBUG
//#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace std::chrono;
#define int long long
#define ll long long
auto start_time = system_clock::now();

//@formatter:off
#ifdef _DEBUG
//区間削除は出来ない
template<class T> struct my_pbds_tree {    set<T> s;    auto begin() { return s.begin(); }    auto end() { return s.end(); }    auto rbegin() { return s.rbegin(); }    auto rend() { return s.rend(); }    auto empty() { return s.empty(); }    auto size() { return s.size(); }    void clear() { s.clear(); }    template<class U> void insert(U v) { s.insert(v); }template<class U> void operator+=(U v) { insert(v); }    template<class F> auto erase(F v) { return s.erase(v); }    template<class U> auto find(U v) { return s.find(v); }    template<class U> auto lower_bound(U v) { return s.lower_bound(v); }    template<class U> auto upper_bound(U v) { return s.upper_bound(v); }    auto find_by_order(ll k) {        auto it = s.begin();        for (ll i = 0; i < k; i++)it++;        return it;    }    auto order_of_key(ll v) {        auto it = s.begin();        ll i=0;        for (;it != s.end() && *it <v ; i++)it++;        return i;    }};
#define pbds(T) my_pbds_tree<T>
//gp_hash_tableでcountを使えないようにするため
template<class T,class U> struct my_unordered_map{    unordered_map<T,U> m;    my_unordered_map(){};    auto begin(){        return m.begin();    }    auto end(){return m.end();}    auto cbegin(){return m.cbegin();}    auto cend(){return m.cend();}    template<class V>auto erase(V v){return m.erase(v);}    void clear(){m.clear();}    /*countは gp_hash_tableに存在しない*/    /*!= m.end()*/    template<class V>auto find(V v){return m.find(v);}    template<class V>auto & operator [](V n) { return m[n] ;}};
#define unordered_map my_unordered_map
#define umapi unordered_map<ll,ll>
#define umapp unordered_map<P,ll>
#define umapip unordered_map<ll,P>

#else
#define unordered_map __gnu_pbds::gp_hash_table
//find_by_order(k) k番目のイテレーター
//order_of_key(k)  k以上が前から何番目か
#define pbds(U) __gnu_pbds::tree<U, __gnu_pbds::null_type, less<U>, __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update>

#define umapi unordered_map<ll,ll,xorshift>
#define umapp unordered_map<P,ll,xorshift>
#define umapip unordered_map<ll,P,xorshift>

#endif
struct xorshift {    static uint64_t splitmix64(uint64_t x) {        x += 0x9e3779b97f4a7c15;        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;        return x ^ (x >> 31);    }    size_t operator()(uint64_t x) const {        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();        return splitmix64(x + FIXED_RANDOM);    }    size_t operator()(std::pair<ll, ll> x) const {        ll v=((x.first) << 32) | x.second;        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();        return splitmix64(v + FIXED_RANDOM);    }};
template<class U, class L> void operator+=(__gnu_pbds::tree<U, __gnu_pbds::null_type, less<U>, __gnu_pbds::rb_tree_tag, __gnu_pbds::tree_order_statistics_node_update> &s, L v) { s.insert(v); }
//衝突対策
#define ws wszzzz

template<class A, class B, class C>struct T2 {A f;B s;C t;T2() { f = 0, s = 0, t = 0; }T2(A f, B s, C t) : f(f), s(s), t(t) {}bool operator<(const T2 &r) const {        return f != r.f ? f < r.f : s != r.s ? s < r.s : t < r.t;        /*return f != r.f ? f > r.f : s != r.s ?n s > r.s : t > r.t; 大きい順 */   }    bool operator>(const T2 &r) const {        return f != r.f ? f > r.f : s != r.s ? s > r.s : t > r.t;        /*return f != r.f ? f > r.f : s != r.s ? s > r.s : t > r.t; 小さい順 */   }    bool operator==(const T2 &r) const {        return f == r.f && s == r.s && t == r.t;    }    bool operator!=(const T2 &r) const {        return f != r.f || s != r.s || t != r.t;    }};
template<class A, class B, class C, class D> struct F2 {    A a;    B b;    C c;    D d;    F2() { a = 0, b = 0, c = 0, d = 0; }    F2(A a, B b, C c, D d) : a(a), b(b), c(c), d(d) {}    bool operator<(const F2 &r) const {        return a != r.a ? a < r.a : b != r.b ? b < r.b : c != r.c ? c < r.c : d < r.d;    /*    return a != r.a ? a > r.a : b != r.b ? b > r.b : c != r.c ? c > r.c : d > r.d;*/    }    bool operator>(const F2 &r) const {        return a != r.a ? a > r.a : b != r.b ? b > r.b : c != r.c ? c > r.c : d > r.d;/*        return a != r.a ? a < r.a : b != r.b ? b < r.b : c != r.c ? c < r.c : d < r.d;*/    }    bool operator==(const F2 &r) const {        return a == r.a && b == r.b && c == r.c && d == r.d;    }    bool operator!=(const F2 &r) const {        return a != r.a || b != r.b || c != r.c || d != r.d;    }    ll operator[](ll i) {        assert(i < 4);        return i == 0 ? a : i == 1 ? b : i == 2 ? c : d;    }};
typedef T2<ll, ll, ll> T;
typedef F2<ll, ll, ll, ll> F;
T mt(ll a, ll b, ll c) {return T(a, b, c);}

//@マクロ省略系 型,構造
#define double long double
#define pow powl
#define ull unsigned long long
using dou = double;
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
#define wh while
#define upd update

#define maxq 1
#define minq -1

#define ZERO(a) memset(a,0,sizeof(a))
#define MINUS(a) memset(a,0xff,sizeof(a))
#define MALLOC(type, len) (type*)malloc((len) * sizeof(type))
#define lam(right) [&](auto& p){return p right;}

//マクロ省略系 コンテナ
using vi = vector<ll>;
using vb = vector<bool>;
using vs = vector<string>;
using vd = vector<double>;
using vc = vector<char>;
using vp = vector<P>;
using vt = vector<T>;

#define vec vector
#define o_vvt(o1, o2, o3, o4, name, ...) name
#define vvt0(t) vec<vec<t>>
#define vvt1(t,a) vec<vec<t>>a
#define vvt2(t,a, b) vec<vec<t>>a(b)
#define vvt3(t,a, b, c) vec<vec<t>> a(b,vec<t>(c))
#define vvt4(t,a, b, c, d) vec<vec<t>> a(b,vec<t>(c,d))

#define vvi(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(ll,__VA_ARGS__)
#define vvb(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(bool,__VA_ARGS__)
#define vvs(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(string,__VA_ARGS__)
#define vvd(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(double,__VA_ARGS__)
#define vvc(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(char,__VA_ARGS__)
#define vvp(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(P,__VA_ARGS__)
#define vvt(...) o_vvt(__VA_ARGS__,vvt4,vvt3,vvt2 ,vvt1,vvt0)(T,__VA_ARGS__)

template<typename T> vector<T> make_v(size_t a) { return vector<T>(a); }
template<typename T, typename... Ts> auto make_v(size_t a, Ts... ts) {return vector<decltype(make_v<T>(ts...))>(a, make_v<T>(ts...));}
#define vni(name, ...) auto name = make_v<ll>(__VA_ARGS__)
#define vnb(name, ...) auto name = make_v<bool>(__VA_ARGS__)
#define vns(name, ...) auto name = make_v<string>(__VA_ARGS__)
#define vnd(name, ...) auto name = make_v<double>(__VA_ARGS__)
#define vnc(name, ...) auto name = make_v<char>(__VA_ARGS__)
#define vnp(name, ...) auto name = make_v<P>(__VA_ARGS__)

#define PQ priority_queue<ll, vector<ll>, greater<ll> >
#define tos to_string
using mapi = map<ll, ll>;
using mapp = map<P, ll>;
using mapd = map<dou, ll>;
using mapc = map<char, ll>;
using maps = map<str, ll>;
using seti = set<ll>;
using setd = set<dou>;
using setc = set<char>;
using sets = set<str>;
using qui = queue<ll>;
#define uset unordered_set
#define useti unordered_set<ll,ll,xorshift>
#define mset multiset
#define mseti multiset<ll>
#define umap unordered_map
#define mmap multimap

template<class T> struct pq {
    priority_queue<T, vector<T>, greater<T> > q;/*小さい順*/    T su = 0;
    void clear() {        q = priority_queue<T, vector<T>, greater<T> >();        su = 0;    }
    void operator+=(T v) {        su += v;        q.push(v);    }
    T sum() { return su; }
    T top() { return q.top(); }
    void pop() {        su -= q.top();        q.pop();    }
    T poll() {        T ret = q.top();        su -= ret;        q.pop();        return ret;    }
    ll size() { return q.size(); }
};
template<class T> struct pqg {
    priority_queue<T> q;/*大きい順*/    T su = 0;
    void clear() {        q = priority_queue<T>();        su = 0;    }
    void operator+=(T v) {        su += v;        q.push(v);    }
    T sum() { return su; }
    T top() { return q.top(); }
    void pop() {        su -= q.top();        q.pop();    }
    T poll() {        T ret = q.top();        su -= ret;        q.pop();        return ret;    }
    ll size() { return q.size(); }
};
#define pqi pq<ll>
#define pqgi pqg<ll>
//マクロ 繰り返し
#define o_rep(o1, o2, o3, o4, name, ...) name
# define rep1(n) for(ll rep1i = 0,rep1lim=n; rep1i < rep1lim ; ++rep1i)
# define rep2(i, n) for(ll i = 0,rep2lim=n; i < rep2lim ; ++i)
#define rep3(i, m, n) for(ll i = m,rep3lim=n; i < rep3lim ; ++i)
#define rep4(i, m, n, ad) for(ll i = m,rep4lim=n; i < rep4lim ; i+= ad)
#define rep(...) o_rep(__VA_ARGS__,rep4,rep3,rep2,rep1)(__VA_ARGS__)

#define rer2(i, n) for(ll i = n; i >= 0 ; i--)
#define rer3(i, m, n) for(ll i = m,rer3lim=n; i >= rer3lim ; i--)
#define rer4(i, m, n, dec) for(ll i = m,rer4lim=n; i >= rer4lim ; i-=dec)
#define rer(...) o_rep(__VA_ARGS__,rer4,rer3,rer2,)(__VA_ARGS__)

#define reps2(i, j, n) for(ll i = 0,reps2lim=n; i < reps2lim ;++i)for(ll j = 0; j < reps2lim ; ++j)
#define reps3(i, j, k, n) for(ll i = 0,reps3lim=n; i < reps3lim ; ++i)for(ll j = 0; j < reps3lim ; ++j)for(ll k = 0; k < reps3lim ; ++k)
#define reps4(i, j, k, l, n) for(ll i = 0,reps4lim=n; i < reps4lim ; ++i)for(ll j = 0; j < reps4lim ; ++j)for(ll k = 0; k < reps4lim ; ++k)for(ll l = 0; l < reps4lim ; ++l)
#define o_reps(o1, o2, o3, o4, o5, name, ...) name
#define reps(...) o_reps(__VA_ARGS__,reps4,reps3,reps2,rep2,)(__VA_ARGS__)

#define repss(i, j, k, a, b, c) for(ll i = 0; i < a ; ++i)for(ll j = 0; j < b ; ++j)for(ll k = 0; k < c ; ++k)
#define repv(i, j, A) rep(i,sz(A))rep(j,sz(A[0]))

#define fora(a, b) for(auto&& a : b)
//インデックスを前後含めて走査
#define fori(i, s, len) for (int i = s, prev = (s == 0) ? len - 1 : s - 1, next = (s == len - 1) ? 0 : s + 1, cou = 0; cou < len; cou++, prev = i, i = next, next = (next == len - 1) ? 0 : next + 1)
//vectorの中身を先頭から見る
#define foriv(i, v, d) int i=0;for (auto prev = d[sz(d)-1],next=d[1],v = d[0]; i < sz(d); i++, prev = v, v = next, next = (i>=sz(d)-1?d[0]:d[i+1]))
#define form(st, l, r) for (auto &&it = st.lower_bound(l); it != st.end() && (*it).fi < r; ++it)
#define forit(st, l, r) for (auto &&it = st.lower_bound(l); it != st.end() && (*it) < r;)

//マクロ 定数
#define k3 1010
#define k4 10101
#define k5 101010
#define k6 1010101
#define k7 10101010
const ll inf = (ll) 1e9 + 100;
const ll linf = (ll) 1e18 + 100;
const char infc = '{';
const string infs = "{";
const double eps = 1e-9;
const double PI = 3.1415926535897932384626433832795029L;

//マクロ省略形 関数等
#define arsz(a) (sizeof(a)/sizeof(a[0]))
#define sz(a) ((ll)(a).size())
#define mp make_pair
#define pb pop_back
#define pf push_front
#define eb emplace_back
#define all(a) (a).begin(),(a).end()
#define rall(a) (a).rbegin(),(a).rend()

constexpr bool ev(ll a) { return !(a & 1); }
constexpr bool od(ll a) { return (a & 1); }
//@拡張系 こう出来るべきというもの

//埋め込み 存在を意識せずに機能を増やされているもの
namespace std {
    template<> class hash<std::pair<signed, signed>> { public:size_t operator()(const std::pair<signed, signed> &x) const { return hash<ll>()(((ll) x.first << 32) | x.second); }};
    template<> class hash<std::pair<ll, ll>> { public:/*大きいllが渡されると、<<32でオーバーフローするがとりあえず問題ないと判断*/size_t operator()(const std::pair<ll, ll> &x) const { return hash<ll>()(((ll) x.first << 32) | x.second); }};
}
//stream まとめ
istream &operator>>(istream &iss, P &a) {    iss >> a.first >> a.second;    return iss;}template<typename T> istream &operator>>(istream &iss, vector<T> &vec) {    for (T &x: vec) iss >> x;    return iss;}template<class T, class U> ostream &operator<<(ostream &os, pair<T, U> p) {    os << p.fi << " " << p.se;    return os;}ostream &operator<<(ostream &os, T p) {    os << p.f << " " << p.s << " " << p.t;    return os;}ostream &operator<<(ostream &os, F p) {    os << p.a << " " << p.b << " " << p.c << " " << p.d;    return os;}template<typename T> ostream &operator<<(ostream &os, vector <T> &vec) {    for (ll i = 0; i < vec.size(); ++i)os << vec[i] << (i + 1 == vec.size() ? "" : " ");    return os;}template<typename T,typename U> ostream &operator<<(ostream &os, vector <pair<T,U>> &vec) {    for (ll i = 0; i < vec.size(); ++i){os << vec[i]; if(i!=vec.size()-1)os<<endl;}   return os;}template<typename T> ostream &operator<<(ostream &os, vector <vector<T>> &vec) {    for (ll i = 0; i < vec.size(); ++i) {        for (ll j = 0; j < vec[i].size(); ++j) { os << vec[i][j] << " "; }        os << endl;    }    return os;}template<typename T, typename U> ostream &operator<<(ostream &os, map<T, U> &m) {    for (auto &&v:m) os << v;    return os;}template<class T> ostream &operator<<(ostream &os, set<T> s) {    fora(v, s) { os << v << " "; }    return os;}template<class T> ostream &operator<<(ostream &os, deque<T> a) {    fora(v, a)os << v << " ";    return os;}ostream &operator<<(ostream &os, vector <vector<char>> &vec) {    rep(h, sz(vec)) {        rep(w, sz(vec[0])) { os << vec[h][w]; }        os << endl;    }    return os;}
template<typename W, typename H> void resize(vector<W> &vec, const H head) { vec.resize(head); }template<typename W, typename H, typename ... T> void resize(vector<W> &vec, const H &head, const T ... tail) {vec.resize(head);for (auto &v: vec)resize(v, tail...);}
template<typename T, typename F> bool all_of2(T &v, F f) { return f(v); }
template<typename T, typename F> bool all_of2(vector<T> &v, F f) {    rep(i, sz(v)) { if (!all_of2(v[i], f))return false; }    return true;}
template<typename T, typename F> bool any_of2(T &v, F f) { return f(v); }
template<typename T, typename F> bool any_of2(vector<T> &v, F f) {    rep(i, sz(v)) { if (any_of2(v[i], f))return true; }    return false;}
template<typename T, typename F> bool none_of2(T &v, F f) { return f(v); }
template<typename T, typename F> bool none_of2(vector<T> &v, F f) {    rep(i, sz(v)) { if (none_of2(v[i], f))return false; }    return true;}
template<typename T, typename F> bool find_if2(T &v, F f) { return f(v); }
template<typename T, typename F> ll find_if2(vector<T> &v, F f) {    rep(i, sz(v)) { if (find_if2(v[i], f))return i; }    return sz(v);}
template<typename T, typename F> bool rfind_if2(T &v, F f) { return f(v); }
template<typename T, typename F> ll rfind_if2(vector<T> &v, F f) {    rer(i, sz(v) - 1) { if (rfind_if2(v[i], f))return i; }    return -1;}
template<class T> bool contains(string &s, const T &v) { return s.find(v) != string::npos; }
template<typename T> bool contains(vector<T> &v, const T &val) { return std::find(v.begin(), v.end(), val) != v.end(); }
template<typename T, typename F> bool contains_if2(vector<T> &v, F f) { return find_if(v.begin(), v.end(), f) != v.end(); }
template<typename T, typename F> ll count_if2(T &v, F f) { return f(v); }
template<typename T, typename F> ll count_if2(vector<T> &vec, F f) {    ll ret = 0;    fora(v, vec)ret += count_if2(v, f);    return ret;}
template<typename T, typename F> void for_each2(T &v, F f) { f(v); }
template<typename T, typename F> void for_each2(vector<T> &vec, F f) { fora(v, vec)for_each2(v, f); }
template<typename W> ll count_od(vector<W> &a) {return count_if2(a,[](ll v){return v&1 ;});}
template<typename W> ll count_ev(vector<W> &a) {return count_if2(a,[](ll v){return !(v&1) ;});}
//削除された要素の数を返す
template<typename T, typename F> int erase_if2(vector<T> &v, F f) {vector<T> nv;int cou = 0;rep(i, sz(v)) {if (f(v[i])) {cou++;} else {nv.push_back(v[i]);}}v = nv;return cou;}
template<typename T, typename F> int erase_if2(vector<vector<T>> &v, F f) {int cou = 0;rep(i, sz(v)) {cou += erase_if2(v[i], f);}return cou;}
#define all_of(a,right) all_of2(a,lam(right))
#define all_of_f(a,f) all_of2(a,f)
#define any_of(a,right) any_of2(a,lam(right))
#define any_of_f(a,f) any_of2(a,f)
#define none_of(a,right) none_of2(a,lam(right))
#define none_of_f(a,f) none_of2(a,f)
#define find_if(a,right) find_if2(a,lam(right))
#define find_if_f(a,f) find_if2(a,f)
#define rfind_if(a,right) rfind_if2(a,lam(right))
#define rfind_if_f(a,f) rfind_if2(a,f)
#define contains_if(a,right) contains_if2(a,lam(right))
#define contains_if_f(a,f) contains_if2(a,f)
#define count_if(a, right) count_if2(a,lam(right))
#define count_if_f(a, f) count_if2(a,f)
#define for_each(a, right) do{fora(v,a){v right;}}while(0)
#define for_each_f(a, f) do{fora(v,a){f(v);}}while(0)
#define erase_if(a,right) erase_if2(a,lam(right))
#define erase_if_f(a,f) erase_if2(a,f)


template<class T, class U> void replace(vector<T> &a, T key, U v) { replace(a.begin(), a.end(), key, v); }
void replace(str &a, char key, str v) { if (v == "")a.erase(remove(all(a), key), a.end()); }
void replace(str &a, char key, char v) { replace(all(a), key, v); }
//keyと同じかどうか01で置き換える
template<class T, class U> void replace(vector<T> &a, U k) { rep(i, sz(a)) a[i] = a[i] == k; }
template<class T, class U> void replace(vector<vector<T >> &a, U k) { rep(i, sz(a))rep(j, sz(a[0])) a[i][j] = a[i][j] == k; }
template<class T> void replace(T &a) { replace(a, '#'); }
void replace(str &a, str key, str v) {stringstream t;ll kn = sz(key);std::string::size_type Pos(a.find(key));ll l = 0;while (Pos != std::string::npos) {t << a.substr(l, Pos - l);t << v;l = Pos + kn;Pos = a.find(key, Pos + kn);}t << a.substr(l, sz(a) - l);a = t.str();}
template<class T> bool includes(vector<T> &a, vector<T> &b) {vi c = a;vi d = b;sort(all(c));sort(all(d));return includes(all(c), all(d));}
template<class T> bool is_permutation(vector<T> &a, vector<T> &b) { return is_permutation(all(a), all(b)); }
template<class T> bool next_permutation(vector<T> &a) { return next_permutation(all(a)); }
void iota(vector<ll> &ve, ll s, ll n) {ve.resize(n);iota(all(ve), s);}
vi iota(ll s, ll len) {vi ve(len);iota(all(ve), s);return ve;}
template<class A, class B> auto vtop(vector<A> &a, vector<B> &b) {    assert(sz(a) == sz(b));    /*stringを0で初期化できない  */  vector<pair<A, B>> res;    rep(i, sz(a))res.eb(a[i], b[i]);return res;}
template<class A, class B> void ptov(vector<pair<A, B>> &p, vector<A> &a, vector<B> &b) {    a.resize(sz(p)), b.resize(sz(p));    rep(i, sz(p))a[i] = p[i].fi, b[i] = p[i].se;}
template<class A, class B, class C> auto vtot(vector<A> &a, vector<B> &b, vector<C> &c) {    assert(sz(a) == sz(b) && sz(b) == sz(c));    vector<T2<A, B, C>> res;    rep(i, sz(a))res.eb(a[i], b[i], c[i]);    return res;}
template<class A, class B, class C, class D> auto vtof(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    assert(sz(a) == sz(b) && sz(b) == sz(c) && sz(c) == sz(d));    vector<F2<A, B, C, D>> res;    rep(i, sz(a))res.eb(a[i], b[i], c[i], d[i]);    return res;}
enum pcomparator { fisi, fisd, fdsi, fdsd, sifi, sifd, sdfi, sdfd };
enum tcomparator {    fisiti, fisitd, fisdti, fisdtd, fdsiti, fdsitd, fdsdti, fdsdtd,    fitisi, fitisd, fitdsi, fitdsd, fdtisi, fdtisd, fdtdsi, fdtdsd,    sifiti, sifitd, sifdti, sifdtd, sdfiti, sdfitd, sdfdti, sdfdtd,    sitifi, sitifd, sitdfi, sitdfd, sdtifi, sdtifd, sdtdfi, sdfdfd,    tifisi, tifisd, tifdsi, tifdsd, tdfisi, tdfisd, tdfdsi, tdfdsd,    tisifi, tisifd, tisdfi, tisdfd, tdsifi, tdsifd, tdsdfi, tdsdfd};
template<class A, class B> void sort(vector<pair<A, B>> &a, pcomparator type) {    typedef pair<A, B> U;    if (type == fisi) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi < r.fi : l.se < r.se; });    else if (type == fisd) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi < r.fi : l.se > r.se; });    else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi > r.fi : l.se < r.se; });    else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.fi != r.fi ? l.fi > r.fi : l.se > r.se; });    else if (type == sifi) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se < r.se : l.fi < r.fi; });    else if (type == sifd) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se < r.se : l.fi > r.fi; });    else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se > r.se : l.fi < r.fi; });    else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.se != r.se ? l.se > r.se : l.fi > r.fi; });};template<class U> void sort(vector<U> &a, pcomparator type) {    if (type == fisi) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s < r.s; });    else if (type == fisd) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s > r.s; });    else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s < r.s; });    else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s > r.s; });    else if (type == sifi) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f < r.f; });    else if (type == sifd) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f > r.f; });    else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f < r.f; });    else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f > r.f; });};template<class A, class B, class C, class D> void sort(vector<F2<A, B, C, D> > &a, pcomparator type) {    typedef F2<A, B, C, D> U;    if (type == fisi) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b < r.b; });    else if (type == fisd) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b > r.b; });    else if (type == fdsi) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b < r.b; });    else if (type == fdsd) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b > r.b; });    else if (type == sifi) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a < r.a; });    else if (type == sifd) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a > r.a; });    else if (type == sdfi) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a < r.a; });    else if (type == sdfd) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a > r.a; });};template<class U> void sort(vector<U> &a, tcomparator type) {    if (type == 0) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s < r.s : l.t < r.t; });    else if (type == 1) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s < r.s : l.t > r.t; });    else if (type == 2) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s > r.s : l.t < r.t; });    else if (type == 3) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.s != r.s ? l.s > r.s : l.t > r.t; });    else if (type == 4) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s < r.s : l.t < r.t; });    else if (type == 5) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s < r.s : l.t > r.t; });    else if (type == 6) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s > r.s : l.t < r.t; });    else if (type == 7) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.s != r.s ? l.s > r.s : l.t > r.t; });    else if (type == 8) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t < r.t : l.s < r.s; });    else if (type == 9) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t < r.t : l.s > r.s; });    else if (type == 10) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t > r.t : l.s < r.s; });    else if (type == 11) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f < r.f : l.t != r.t ? l.t > r.t : l.s > r.s; });    else if (type == 12) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t < r.t : l.s < r.s; });    else if (type == 13) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t < r.t : l.s > r.s; });    else if (type == 14) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t > r.t : l.s < r.s; });    else if (type == 15) sort(all(a), [&](U l, U r) { return l.f != r.f ? l.f > r.f : l.t != r.t ? l.t > r.t : l.s > r.s; });    else if (type == 16) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f < r.f : l.t < r.t; });    else if (type == 17) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f < r.f : l.t > r.t; });    else if (type == 18) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f > r.f : l.t < r.t; });    else if (type == 19) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.f != r.f ? l.f > r.f : l.t > r.t; });    else if (type == 20) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f < r.f : l.t < r.t; });    else if (type == 21) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f < r.f : l.t > r.t; });    else if (type == 22) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f > r.f : l.t < r.t; });    else if (type == 23) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.f != r.f ? l.f > r.f : l.t > r.t; });    else if (type == 24) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t < r.t : l.f < r.f; });    else if (type == 25) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t < r.t : l.f > r.f; });    else if (type == 26) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t > r.t : l.f < r.f; });    else if (type == 27) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s < r.s : l.t != r.t ? l.t > r.t : l.f > r.f; });    else if (type == 28) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t < r.t : l.f < r.f; });    else if (type == 29) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t < r.t : l.f > r.f; });    else if (type == 30) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t > r.t : l.f < r.f; });    else if (type == 31) sort(all(a), [&](U l, U r) { return l.s != r.s ? l.s > r.s : l.t != r.t ? l.t > r.t : l.f > r.f; });    else if (type == 32) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f < r.f : l.s < r.s; });    else if (type == 33) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f < r.f : l.s > r.s; });    else if (type == 34) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f > r.f : l.s < r.s; });    else if (type == 35) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.f != r.f ? l.f > r.f : l.s > r.s; });    else if (type == 36) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f < r.f : l.s < r.s; });    else if (type == 37) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f < r.f : l.s > r.s; });    else if (type == 38) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f > r.f : l.s < r.s; });    else if (type == 39) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.f != r.f ? l.f > r.f : l.s > r.s; });    else if (type == 40) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s < r.s : l.f < r.f; });    else if (type == 41) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s < r.s : l.f > r.f; });    else if (type == 42) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s > r.s : l.f < r.f; });    else if (type == 43) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t < r.t : l.s != r.s ? l.s > r.s : l.f > r.f; });    else if (type == 44) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s < r.s : l.f < r.f; });    else if (type == 45) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s < r.s : l.f > r.f; });    else if (type == 46) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s > r.s : l.f < r.f; });    else if (type == 47) sort(all(a), [&](U l, U r) { return l.t != r.t ? l.t > r.t : l.s != r.s ? l.s > r.s : l.f > r.f; });}template<class A, class B, class C, class D> void sort(vector<F2<A, B, C, D>> &a, tcomparator type) {    typedef F2<A, B, C, D> U;    if (type == 0) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b < r.b : l.c < r.c; });    else if (type == 1) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b < r.b : l.c > r.c; });    else if (type == 2) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b > r.b : l.c < r.c; });    else if (type == 3) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.b != r.b ? l.b > r.b : l.c > r.c; });    else if (type == 4) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b < r.b : l.c < r.c; });    else if (type == 5) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b < r.b : l.c > r.c; });    else if (type == 6) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b > r.b : l.c < r.c; });    else if (type == 7) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.b != r.b ? l.b > r.b : l.c > r.c; });    else if (type == 8) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c < r.c : l.b < r.b; });    else if (type == 9) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c < r.c : l.b > r.b; });    else if (type == 10) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c > r.c : l.b < r.b; });    else if (type == 11) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a < r.a : l.c != r.c ? l.c > r.c : l.b > r.b; });    else if (type == 12) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c < r.c : l.b < r.b; });    else if (type == 13) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c < r.c : l.b > r.b; });    else if (type == 14) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c > r.c : l.b < r.b; });    else if (type == 15) sort(all(a), [&](U l, U r) { return l.a != r.a ? l.a > r.a : l.c != r.c ? l.c > r.c : l.b > r.b; });    else if (type == 16) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a < r.a : l.c < r.c; });    else if (type == 17) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a < r.a : l.c > r.c; });    else if (type == 18) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a > r.a : l.c < r.c; });    else if (type == 19) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.a != r.a ? l.a > r.a : l.c > r.c; });    else if (type == 20) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a < r.a : l.c < r.c; });    else if (type == 21) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a < r.a : l.c > r.c; });    else if (type == 22) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a > r.a : l.c < r.c; });    else if (type == 23) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.a != r.a ? l.a > r.a : l.c > r.c; });    else if (type == 24) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c < r.c : l.a < r.a; });    else if (type == 25) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c < r.c : l.a > r.a; });    else if (type == 26) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c > r.c : l.a < r.a; });    else if (type == 27) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b < r.b : l.c != r.c ? l.c > r.c : l.a > r.a; });    else if (type == 28) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c < r.c : l.a < r.a; });    else if (type == 29) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c < r.c : l.a > r.a; });    else if (type == 30) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c > r.c : l.a < r.a; });    else if (type == 31) sort(all(a), [&](U l, U r) { return l.b != r.b ? l.b > r.b : l.c != r.c ? l.c > r.c : l.a > r.a; });    else if (type == 32) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a < r.a : l.b < r.b; });    else if (type == 33) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a < r.a : l.b > r.b; });    else if (type == 34) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a > r.a : l.b < r.b; });    else if (type == 35) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.a != r.a ? l.a > r.a : l.b > r.b; });    else if (type == 36) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a < r.a : l.b < r.b; });    else if (type == 37) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a < r.a : l.b > r.b; });    else if (type == 38) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a > r.a : l.b < r.b; });    else if (type == 39) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.a != r.a ? l.a > r.a : l.b > r.b; });    else if (type == 40) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b < r.b : l.a < r.a; });    else if (type == 41) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b < r.b : l.a > r.a; });    else if (type == 42) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b > r.b : l.a < r.a; });    else if (type == 43) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c < r.c : l.b != r.b ? l.b > r.b : l.a > r.a; });    else if (type == 44) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b < r.b : l.a < r.a; });    else if (type == 45) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b < r.b : l.a > r.a; });    else if (type == 46) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b > r.b : l.a < r.a; });    else if (type == 47) sort(all(a), [&](U l, U r) { return l.c != r.c ? l.c > r.c : l.b != r.b ? l.b > r.b : l.a > r.a; });}

void sort(string &a) { sort(all(a)); }
template<class T> void sort(vector<T> &a) { sort(all(a)); }
//P l, P rで f(P) の形で渡す
template<class U, class F> void sort(vector<U> &a, F f) { sort(all(a), [&](U l, U r) { return f(l) < f(r); }); };
template<class T> void rsort(vector<T> &a) { sort(all(a), greater<T>()); };
template<class U, class F> void rsort(vector<U> &a, F f) { sort(all(a), [&](U l, U r) { return f(l) > f(r); }); };
//F = T<T>
//例えばreturn p.fi + p.se;
template<class A, class B> void sortp(vector<A> &a, vector<B> &b) {    auto c = vtop(a, b);    sort(c);    rep(i, sz(a)) a[i] = c[i].fi, b[i] = c[i].se;}template<class A, class B, class F> void sortp(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    sort(c, f);    rep(i, sz(a)) a[i] = c[i].fi, b[i] = c[i].se;}template<class A, class B> void rsortp(vector<A> &a, vector<B> &b) {    auto c = vtop(a, b);    rsort(c);    rep(i, sz(a))a[i] = c[i].first, b[i] = c[i].second;}template<class A, class B, class F> void rsortp(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    rsort(c, f);    rep(i, sz(a))a[i] = c[i].first, b[i] = c[i].second;}
template<class A, class B, class C> void sortt(vector<A> &a, vector<B> &b, vector<C> &c) {    auto d = vtot(a, b, c);    sort(d);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C, class F> void sortt(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    sort(d, f);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C> void rsortt(vector<A> &a, vector<B> &b, vector<C> &c) {    auto d = vtot(a, b, c);    rsort(d);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C, class F> void rsortt(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    rsort(d, f);    rep(i, sz(a)) a[i] = d[i].f, b[i] = d[i].s, c[i] = d[i].t;}
template<class A, class B, class C, class D> void sortf(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    auto e = vtof(a, b, c, d);    sort(e);    rep(i, sz(a)) a[i] = e[i].a, b[i] = e[i].b, c[i] = e[i].c, d[i] = e[i].d;}
template<class A, class B, class C, class D> void rsortf(vector<A> &a, vector<B> &b, vector<C> &c, vector<D> &d) {    auto e = vtof(a, b, c, d);    rsort(e);    rep(i, sz(a)) a[i] = e[i].a, b[i] = e[i].b, c[i] = e[i].c, d[i] = e[i].d;}
//sortindex 元のvectorはソートしない
template<class T> vi sorti(vector<T> &a) {    auto b = a;    vi ind = iota(0, sz(a));    sortp(b, ind);    return ind;}/*indexの分で型が変わるためpcomparatorが必要*/template<class T> vi sorti(vector<T> &a, pcomparator f) {    auto b = a;    vi ind = iota(0, sz(a));    sortp(b, ind, f);    return ind;}template<class T, class F> vi sorti(vector<T> &a, F f) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(a[x]) < f(a[y]); });    return ind;}template<class T> vi rsorti(vector<T> &a) {    auto b = a;    vi ind = iota(0, sz(a));    rsortp(b, ind);    return ind;}template<class T, class F> vi rsorti(vector<T> &a, F f) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(a[x]) > f(a[y]); });    return ind;}template<class A, class B, class F> vi sortpi(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(c[x]) < f(c[y]); });    return ind;}template<class A, class B> vi sortpi(vector<A> &a, vector<B> &b, pcomparator f) {    vi ind = iota(0, sz(a));    auto c = a;    auto d = b;    sortt(c, d, ind, f);    return ind;}template<class A, class B> vi sortpi(vector<A> &a, vector<B> &b) { return sortpi(a, b, fisi); };template<class A, class B, class F> vi rsortpi(vector<A> &a, vector<B> &b, F f) {    auto c = vtop(a, b);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(c[x]) > f(c[y]); });    return ind;}template<class A, class B> vi rsortpi(vector<A> &a, vector<B> &b) { return sortpi(a, b, fdsd); };template<class A, class B, class C, class F> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(d[x]) < f(d[y]); });    return ind;}template<class A, class B, class C> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c, pcomparator f) {    vi ind = iota(0, sz(a));    auto d = vtof(a, b, c, ind);    sort(d, f);    rep(i, sz(a))ind[i] = d[i].d;    return ind;}template<class A, class B, class C> vi sortti(vector<A> &a, vector<B> &b, vector<C> &c) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) {        if (a[x] == a[y]) {            if (b[x] == b[y])return c[x] < c[y];            else return b[x] < b[y];        } else {            return a[x] < a[y];        }    });    return ind;}template<class A, class B, class C, class F> vi rsortti(vector<A> &a, vector<B> &b, vector<C> &c, F f) {    auto d = vtot(a, b, c);    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) { return f(d[x]) > f(d[y]); });    return ind;}template<class A, class B, class C> vi rsortti(vector<A> &a, vector<B> &b, vector<C> &c) {    vi ind = iota(0, sz(a));    sort(all(ind), [&](ll x, ll y) {        if (a[x] == a[y]) {            if (b[x] == b[y])return c[x] > c[y];            else return b[x] > b[y];        } else {            return a[x] > a[y];        }    });    return ind;}
template<class T> void sort2(vector<vector<T >> &a) { for (ll i = 0, n = a.size(); i < n; ++i)sort(a[i]); }
template<class T> void rsort2(vector<vector<T >> &a) { for (ll i = 0, n = a.size(); i < n; ++i)rsort(a[i]); }

template<typename A, size_t N, typename T> void fill(A (&a)[N], const T &v) { rep(i, N)a[i] = v; }template<typename A, size_t N, size_t O, typename T> void fill(A (&a)[N][O], const T &v) { rep(i, N)rep(j, O)a[i][j] = v; }template<typename A, size_t N, size_t O, size_t P, typename T> void fill(A (&a)[N][O][P], const T &v) { rep(i, N)rep(j, O)rep(k, P)a[i][j][k] = v; }template<typename A, size_t N, size_t O, size_t P, size_t Q, typename T> void fill(A (&a)[N][O][P][Q], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)a[i][j][k][l] = v; }template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, typename T> void fill(A (&a)[N][O][P][Q][R], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)a[i][j][k][l][m] = v; }template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S, typename T> void fill(A (&a)[N][O][P][Q][R][S], const T &v) { rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)rep(n, S)a[i][j][k][l][m][n] = v; }
template<typename W, typename T>void fill(W &xx, const T vall) {    xx = vall;}template<typename W, typename T>void fill(vector<W> &vecc, const T vall) {    for (auto &&vx     : vecc)fill(vx, vall);}
template<typename W,typename T>void fill(vector<W> &xx,const T v,ll len) {rep(i, len)xx[i]=v;}
template<typename W,typename T>void fill(vector<vector<W>> &xx,const T v,ll lh,ll lw) {rep(i, lh)rep(j,lw)xx[i][j]=v;}
template<class T,class U>void fill(vector<T> &a,U val,vi& ind) {fora(v,ind)a[v]=val;}

template<typename A, size_t N> A sum(A (&a)[N]) {    A res = 0;    rep(i, N)res += a[i];    return res;}template<typename A, size_t N, size_t O> A sum(A (&a)[N][O]) {    A res = 0;    rep(i, N)rep(j, O)res += a[i][j];    return res;}template<typename A, size_t N, size_t O, size_t P> A sum(A (&a)[N][O][P]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)res += a[i][j][k];    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q> A sum(A (&a)[N][O][P][Q]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)res += a[i][j][k][l];    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R> A sum(A (&a)[N][O][P][Q][R]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)res += a[i][j][k][l][m];    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S> A sum(A (&a)[N][O][P][Q][R][S]) {    A res = 0;    rep(i, N)rep(j, O)rep(k, P)rep(l, Q)rep(m, R)rep(n, S)res += a[i][j][k][l][m][n];    return res;}
//@汎用便利関数 入力
ll in() {ll ret;cin >> ret;return ret;}
string sin() {string ret;cin >> ret;return ret;}
template<class T>  void in(T &head) { cin >> head; }template<class T, class... U>  void in(T &head, U &... tail) {cin >> head;in(tail...);}

#define o_din(o1, o2, o3, o4, o5, o6, name, ...) name
#define din1(a) ll a;cin>>a
#define din2(a, b) ll a,b;cin>>a>> b
#define din3(a, b, c) ll a,b,c;cin>>a>>b>>c
#define din4(a, b, c, d) ll a,b,c,d;cin>>a>>b>>c>>d
#define din5(a, b, c, d, e) ll a,b,c,d,e;cin>>a>>b>>c>>d>>e
#define din6(a, b, c, d, e, f) ll a,b,c,d,e,f;cin>>a>>b>>c>>d>>e>>f
#define din(...) o_din(__VA_ARGS__,din6,din5,din4,din3,din2 ,din1)(__VA_ARGS__)

#define o_dins(o1, o2, o3, o4, o5, o6, name, ...) name
#define dins1(a) str a;cin>>a
#define dins2(a, b) str a,b;cin>>a>> b
#define dins3(a, b, c) str a,b,c;cin>>a>>b>>c
#define dins4(a, b, c, d) str a,b,c,d;cin>>a>>b>>c>>d
#define dins5(a, b, c, d, e) str a,b,c,d,e;cin>>a>>b>>c>>d>>e
#define dins6(a, b, c, d, e, f) str a,b,c,d,e,f;cin>>a>>b>>c>>d>>e>>f
#define dins(...) o_dins(__VA_ARGS__,dins6,dins5,dins4,dins3,dins2 ,dins1)(__VA_ARGS__)

#define o_dind(o1, o2, o3, o4, name, ...) name
#define din1d(a) din1(a);a--
#define din2d(a, b) din2(a,b);a--,b--
#define din3d(a, b, c) din3(a,b,c);a--,b--,c--
#define din4d(a, b, c, d) din4(a,b,c,d);a--,b--,c--,d--
#define dind(...) o_dind(__VA_ARGS__,din4d,din3d,din2d ,din1d)(__VA_ARGS__)

template<class T> void out2(T &&head) { cout << head; }
template<class T, class... U> void out2(T &&head, U &&... tail) {    cout << head << " ";    out2(tail...);}
template<class T, class... U> void out(T &&head, U &&... tail) {    cout << head << " ";    out2(tail...);    cout << "" << endl;}
template<class T> void out(T &&head) {    cout << head  << endl;}
void out() {    cout << ""  << endl;}


#ifdef _DEBUG
template<class T> void err2(T &&head) { cerr << head; }
template<class T, class... U> void err2(T &&head, U &&... tail) {    cerr << head << " ";    err2(tail...);}
template<class T, class... U> void err(T &&head, U &&... tail) {    cerr << head << " ";    err2(tail...);    cerr << "" << endl;}
template<class T> void err(T &&head) {    cerr << head  << endl;}
void err() {    cerr << ""  << endl;}
template<class T> string out_m2(vector<T> &a, ll W = inf) {stringstream ss;    if (W == inf)W = min(sz(a), 12ll);   if(sz(a)==0)return ss.str();   rep(i, W) { ss << a[i] << " "; }     return ss.str();}
template<class T> string out_m2(vector<vector<T> > &a, ll H = inf, ll W = inf, int key = -1) {H = min({H, sz(a), 12ll});W = min({W, sz(a[0]), 12ll});stringstream ss;ss << endl;if (key == -1)ss << " *|"; else ss << " " << key << "|";rep(w, W)ss << std::right << std::setw(4) << w;ss << "" << endl;rep(w, W * 4 + 3)ss << "_";ss << "" << endl;rep(h, H) {ss << std::right << std::setw(2) << h << "|";rep(w, min(sz(a[h]),12ll)) { if (abs(a[h][w]) == linf) ss << "   e" << ""; else ss << std::right << std::setw(4) << a[h][w]; }ss << "" << endl;}return ss.str();}
template<class T> string out_m2(vector<vector<vector<T> > > &a, ll H = inf, ll W = inf, ll U = inf) {stringstream ss;    if (H == inf)H = 5;    H = min(H, sz(a));    rep(i, H) {        ss << endl;        ss << out_m2(a[i], W, U, i);    }       return ss.str();}
string out_m2(int a) {stringstream ss;ss << a;return ss.str();}
template<class T> string out_m2(T &a) {stringstream ss;ss << a;return ss.str();}

template<class T> string out_m(vector<T> &a, ll W = inf) {stringstream ss;    if (W == inf)W = min(sz(a), 12ll);   if(sz(a)==0)return ss.str();   rep(i, W) { ss << a[i] << " "; }    ss << "" << endl;    return ss.str();}
template<class T> string out_m(vector<vector<T> > &a, ll H = inf, ll W = inf, int key = -1) {H = min({H, sz(a), 12ll});W = min({W, sz(a[0]), 12ll});stringstream ss;ss << endl;if (key == -1)ss << " *|"; else ss << " " << key << "|";rep(w, W)ss << std::right << std::setw(4) << w;ss << "" << endl;rep(w, W * 4 + 3)ss << "_";ss << "" << endl;rep(h, H) {ss << std::right << std::setw(2) << h << "|";rep(w, min(sz(a[h]),12ll)) { if (abs(a[h][w]) == linf) ss << "   e" << ""; else ss << std::right << std::setw(4) << a[h][w]; }ss << "" << endl;}ss << endl;return ss.str();}
template<class T> string out_m(vector<vector<vector<T> > > &a, ll H = inf, ll W = inf, ll U = inf) {stringstream ss;    if (H == inf)H = 5;    H = min(H, sz(a));    rep(i, H) {        ss << endl;        ss << out_m(a[i], W, U, i);    }    ss << endl;    return ss.str();}
string out_m(int a) {stringstream ss;ss << a << endl;return ss.str();}
template<class T> string out_m(T &a) {stringstream ss;ss << a << endl;return ss.str();}
template<class T> void outv(vector<T> &a, ll W=inf) {cout << out_m(a,W) << endl;}
template<class T> void outv(vector<vector<T> > &a, ll H = linf, ll W = linf,int key=-1) {    cout << out_m(a,H,W,key) << endl;}
template<class T> void outv(vector<vector<vector<T> > > &a, ll H = linf, ll W = linf,ll U = linf) {cout << out_m(a,H,W,U)<< endl;}
#else
template<class T> void outv(vector<T> &a, ll W = inf) {
    rep(i, min(W, sz(a))) { cout << a[i] << " "; }
    cout << "" << endl;
}
template<class T> void outv(vector<vector<T> > &a, ll H = linf, ll W = linf, int key = -1) { rep(i, min(H, sz(a))) { outv(a[i], W); }}
template<class T> void outv(vector<vector<vector<T> > > &a, ll H = linf, ll W = linf, ll U = linf) { ; }
#define err(...);
#endif
template<class T> void outl(vector<T> &a, int n = inf) { rep(i, min(n, sz(a)))cout << a[i] << endl; }
//テーブルをスペースなしで出力
template<class T> void outt(vector<vector<T>>&a){    rep(i,sz(a)){        rep(j,sz(a[i])){            cout<<a[i][j];        }        cout<<endl;    }}
//int型をbit表記で出力
void outb(int a){cout << bitset<20>(a) << endl;}
template<class T> void na(vector<T> &a, ll n) {    a.resize(n);    rep(i, n)cin >> a[i];}
#define dna(a, n) vi a(n); rep(dnai,n) cin >> a[dnai];
#define dnad(a, n) vi a(n); rep(dnai,n) cin >> a[dnai],a[dnai]--;
template<class T> void nao(vector<T> &a, ll n) {    a.resize(n + 1);    a[0] = 0;    rep(i, n)cin >> a[i + 1];}
template<class T> void naod(vector<T> &a, ll n) {    a.resize(n + 1);    a[0] = 0;    rep(i, n)cin >> a[i + 1], a[i + 1]--;}
template<class T> void nad(vector<T> &a, ll n) {    a.resize(n);    rep(i, n)cin >> a[i], a[i]--;}
template<class T, class U> void na2(vector<T> &a, vector<U> &b, ll n) {    a.resize(n);    b.resize(n);    rep(i, n)cin >> a[i] >> b[i];}
#define dna2(a, b, n) vi a(n),b(n);rep(dna2i, n)cin >> a[dna2i] >> b[dna2i];
template<class T, class U> void nao2(vector<T> &a, vector<U> &b, ll n) {    a.resize(n + 1);    b.resize(n + 1);    a[0] = b[0] = 0;    rep(i, n)cin >> a[i + 1] >> b[i + 1];}
#define dna2d(a, b, n) vi a(n),b(n);rep(dna2di, n){cin >> a[dna2di] >> b[dna2di];a[dna2di]--,b[dna2di]--;}
template<class T, class U> void na2d(vector<T> &a, vector<U> &b, ll n) {    a.resize(n);    b.resize(n);    rep(i, n)cin >> a[i] >> b[i], a[i]--, b[i]--;}
template<class T, class U, class W> void na3(vector<T> &a, vector<U> &b, vector<W> &c, ll n) {    a.resize(n);    b.resize(n);    c.resize(n);    rep(i, n)cin >> a[i] >> b[i] >> c[i];}
#define dna3(a, b, c, n) vi a(n),b(n),c(n);   rep(dna3i, n)cin >> a[dna3i] >> b[dna3i] >> c[dna3i];
template<class T, class U, class W> void na3d(vector<T> &a, vector<U> &b, vector<W> &c, ll n) {    a.resize(n);    b.resize(n);    c.resize(n);    rep(i, n)cin >> a[i] >> b[i] >> c[i], a[i]--, b[i]--, c[i]--;}
#define dna3d(a, b, c, n) vi a(n),b(n),c(n);  rep(dna3di, n){cin >> a[dna3di] >> b[dna3di] >> c[dna3di];a[dna3di]--,b[dna3di]--,c[dna3di]--;}
template<class T, class U, class W, class X> void na4(vector<T> &a, vector<U> &b, vector<W> &c, vector<X> &d, ll n) {a.resize(n);    b.resize(n);    c.resize(n);    d.resize(n);    rep(i, n)cin >> a[i] >> b[i] >> c[i] >> d[i];}
#define dna4(a, b, c, d, n) vi a(n),b(n),c(n),d(n);   rep(dna4i, n)cin >> a[dna4i] >> b[dna4i] >> c[dna4i]>>d[dna4i];
#define dna4d(a, b, c, d, n) vi a(n),b(n),c(n),d(n);   rep(dna4i, n)cin >> a[dna4i] >> b[dna4i] >> c[dna4i]>>d[dna4i],--a[dna4i] ,-- b[dna4i],-- c[dna4i],--d[dna4i];
#define nt(a, h, w) resize(a,h,w);rep(nthi,h)rep(ntwi,w) cin >> a[nthi][ntwi];
#define ntd(a, h, w) resize(a,h,w);rep(ntdhi,h)rep(ntdwi,w) cin >> a[ntdhi][ntdwi], a[ntdhi][ntdwi]--;
#define ntp(a, h, w) resize(a,h+2,w+2);fill(a,'#');rep(ntphi,1,h+1)rep(ntpwi,1,w+1) cin >> a[ntphi][ntpwi];
//デバッグ
#define sp << " " <<

#define debugName(VariableName) # VariableName

#define deb1(x)  debugName(x)<<" = "<<out_m2(x)
#define deb2(x, ...) deb1(x) <<", "<< deb1(__VA_ARGS__)
#define deb3(x, ...) deb1(x) <<", "<< deb2(__VA_ARGS__)
#define deb4(x, ...) deb1(x) <<", "<< deb3(__VA_ARGS__)
#define deb5(x, ...) deb1(x) <<", "<< deb4(__VA_ARGS__)
#define deb6(x, ...) deb1(x) <<", "<< deb5(__VA_ARGS__)
#define deb7(x, ...) deb1(x) <<", "<< deb6(__VA_ARGS__)
#define deb8(x, ...) deb1(x) <<", "<< deb7(__VA_ARGS__)
#define deb9(x, ...) deb1(x) <<", "<< deb8(__VA_ARGS__)
#define deb10(x, ...) deb1(x) <<", "<< deb9(__VA_ARGS__)

#define o_ebug(o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, name, ...) name

#ifdef _DEBUG
#define deb(...)  cerr<< o_ebug(__VA_ARGS__,deb10,deb9,deb8,deb7,deb6,deb5,deb4,deb3,deb2,deb1)(__VA_ARGS__) <<endl
#else
#define deb(...) ;
#endif


#define debugline(x) cerr << x << " " << "(L:" << __LINE__ << ")" << '\n'


//@formatter:off
//よく使うクラス、構造体
struct unionfind {
    vector<ll> par;
    vector<ll> siz;
    vector<ll> es;
    ll n, trees;//連結グループの数(親の種類)
    unionfind(ll n) : n(n), trees(n) {        par.resize(n);        siz.resize(n);        es.resize(n);        for (ll i = 0; i < n; i++) {            par[i] = i;            siz[i] = 1;        }    }
    ll root(ll x) { if (par[x] == x) { return x; } else { return par[x] = root(par[x]); }}
    ll operator()(ll x){return root(x);}
    bool unite(ll x, ll y) {
        x = root(x);
        y = root(y);
        es[x]++;
        if (x == y) return false;
        if (siz[x] > siz[y]) swap(x, y);
        trees--;
        par[x] = y;
        siz[y] += siz[x];
        es[y] += es[x];
        return true;
    }
    bool same(ll x, ll y) { return root(x) == root(y); }
    ll size(ll x) { return siz[root(x)]; }
    ll esize(ll x) { return es[root(x)]; }
    vi sizes(){        vi cou(n);        vi ret;        ret.reserve(n);        rep(i, n){            cou[root (i)]++;        }        rep(i, n){            if(cou[i])ret.push_back(cou[i]);        }        return ret;    }
    //つながりを無向グラフと見なし、xが閉路に含まれるか判定
    bool close(ll x) { return esize(x) >= size(x); }
    vec<vi> sets() {        vi ind(n, -1);        ll i = 0;        vvi(res, trees);        rep(j, n) {            ll r = root(j);            if (ind[r] == -1)ind[r] = i++;            res[ind[r]].push_back(j);        }        rep(i, trees) {            ll r = root(res[i][0]);            if (res[i][0] == r)continue;            rep(j, 1, sz(res[i])) {                if (res[i][j] == r) {                    swap(res[i][0], res[i][j]);                    break;                }            }        }        return res;    }
};//@formatter:off


using bll =__int128;
using u32 = unsigned;
using u64 = unsigned long long;
using u128 = __uint128_t;

std::ostream &operator<<(std::ostream &dest, __int128_t value) {    std::ostream::sentry s(dest);    if (s) {        __uint128_t tmp = value < 0 ? -value : value;        char buffer[128];        char *d = std::end(buffer);        do {            --d;            *d = "0123456789"[tmp % 10];            tmp /= 10;        } while (tmp != 0);        if (value < 0) {            --d;            *d = '-';        }        ll len = std::end(buffer) - d;        if (dest.rdbuf()->sputn(d, len) != len) { dest.setstate(std::ios_base::badbit); }    }    return dest;}
//__int128 toi128(string &s) {    __int128 ret = 0;    for (ll i = 0; i < s.length(); ++i)        if ('0' <= s[i] && s[i] <= '9')            ret = 10 * ret + s[i] - '0';    return ret;}


//エラー
void ole() {
#ifdef _DEBUG
    debugline("ole");    exit(0);
#endif
    string a = "a";    rep(i, 30)a += a;    rep(i, 1 << 17)cout << a << endl;    cout << "OLE 出力長制限超過" << endl;    exit(0);}
void re() {    assert(0 == 1);    exit(0);}
void tle() { while (inf)cout << inf << endl; }

//便利関数

//テスト用
char ranc() { return (char) ('a' + rand() % 26); }
ll rand(ll min, ll max) {    assert(min <= max);    if (min >= 0 && max >= 0) { return rand() % (max + 1 - min) + min; } else if (max < 0) { return -rand(-max, -min); } else { if (rand() % 2) { return rand(0, max); } else { return -rand(0, -min); }}}
vi ranv(ll n, ll min, ll max) {    vi v(n);    rep(i, n)v[i] = rand(min, max);    return v;}
str ransu(ll n) {    str s;    rep(i, n)s += (char) rand('A', 'Z');    return s;}
str ransl(ll n) {    str s;    rep(i, n)s += (char) rand('a', 'z');    return s;}
//単調増加
vi ranvinc(ll n, ll min, ll max) {    vi v(n);    bool bad = 1;    while (bad) {        bad = 0;        v.resize(n);        rep(i, n) {            if (i && min > max - v[i - 1]) {                bad = 1;                break;            }            if (i)v[i] = v[i - 1] + rand(min, max - v[i - 1]); else v[i] = rand(min, max);        }    }    return v;}
//便利 汎用

void ranvlr(ll n, ll min, ll max, vi &l, vi &r) {    l.resize(n);    r.resize(n);    rep(i, n) {        l[i] = rand(min, max);        r[i] = l[i] + rand(0, max - l[i]);    }}
vp run_length(vi &a) {    vp ret;    ret.eb(a[0], 1);    rep(i, 1, sz(a)) { if (ret.back().fi == a[i]) { ret.back().se++; } else { ret.eb(a[i], 1); }}    return ret;}
vector<pair<char, ll>> run_length(string &a) {    vector<pair<char, ll>> ret;    ret.eb(a[0], 1);    rep(i, 1, sz(a)) { if (ret.back().fi == a[i]) { ret.back().se++; } else { ret.eb(a[i], 1); }}    return ret;}
template<class F> ll mgr(ll ok, ll ng, F f) {    if (ok < ng)        while (ng - ok > 1) {            ll mid = (ok + ng) / 2;            if (f(mid))ok = mid; else ng = mid;        }    else        while (ok - ng > 1) {            ll mid = (ok + ng) / 2;            if (f(mid))ok = mid; else ng = mid;        }    return ok;}
//strを整数として比較
string smax(str &a, str b) {    if (sz(a) < sz(b)) { return b; }    else if (sz(a) > sz(b)) { return a; }    else if (a < b)return b;    else return a;}
//strを整数として比較
string smin(str &a, str b) {    if (sz(a) > sz(b)) { return b; }    else if (sz(a) < sz(b)) { return a; }    else if (a > b)return b;    else return a;}
template<typename W, typename T> ll find(vector<W> &a, const T key) {    rep(i, sz(a))if (a[i] == key)return i;    return -1;}
template<typename W, typename T> P find(vector<vector<W >> &a, const T key) {    rep(i, sz(a))rep(j, sz(a[0]))if (a[i][j] == key)return mp(i, j);    return mp(-1, -1);}
template<typename W, typename U> T find(vector<vector<vector<W >>> &a, const U key) {    rep(i, sz(a))rep(j, sz(a[0]))rep(k, sz(a[0][0]))if (a[i][j][k] == key)return mt(i, j, k);    return mt(-1, -1, -1);}

template<typename W, typename T> ll count2(W &a, const T k) { return a == k; }
template<typename W, typename T> ll count2(vector<W> &a, const T k) {    ll ret = 0;    fora(v, a)ret += count2(v, k);    return ret;}
template<typename W, typename T> ll count(vector<W> &a, const T k) {    ll ret = 0;    fora(v, a)ret += count2(v, k);    return ret;}
ll count(str &a, str k) {    ll ret = 0, len = k.length();    auto pos = a.find(k);    while (pos != string::npos)pos = a.find(k, pos + len), ++ret;    return ret;}
vi count(str &a) {    vi cou(26);    char c = 'a';    if ('A' <= a[0] && a[0] <= 'Z')c = 'A';    rep(i, sz(a))++cou[a[i] - c];    return cou;}
#define couif count_if
//algorythm

ll rev(ll a) {    ll res = 0;    while (a) {        res *= 10;        res += a % 10;        a /= 10;    }    return res;}
template<class T> void rev(vector<T> &a) { reverse(all(a)); }
template<class U> void rev(vector<vector<U>> &a) {    vector<vector<U> > b(sz(a[0]), vector<U>(sz(a)));    rep(h, sz(a)) rep(w, sz(a[0]))b[w][h] = a[h][w];    a = b;}
void  rev(string &a) { reverse(all(a)); }
constexpr ll p10[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000ll, 100000000000ll, 1000000000000ll, 10000000000000ll, 100000000000000ll, 1000000000000000ll, 10000000000000000ll, 100000000000000000ll, 1000000000000000000ll};

ll get(ll a, ll keta) { return (a / (ll) pow(10, keta)) % 10; }
ll keta(ll v) { if (v < p10[9]) { if (v < p10[4]) { if (v < p10[2]) { if (v < p10[1]) return 1; else return 2; } else { if (v < p10[3]) return 3; else return 4; }} else { if (v < p10[7]) { if (v < p10[5]) return 5; else if (v < p10[6])return 6; else return 7; } else { if (v < p10[8])return 8; else return 9; }}} else { if (v < p10[13]) { if (v < p10[11]) { if (v < p10[10]) return 10; else return 11; } else { if (v < p10[12]) return 12; else return 13; }} else { if (v < p10[15]) { if (v < p10[14]) return 14; else if (v < p10[15])return 15; else return 16; } else { if (v < p10[17])return 17; else return 18; }}}}
ll dsum(ll v,ll sin=10) {    ll ret = 0;    for (; v; v /= sin)ret += v % sin;    return ret;}

ll mask10(ll v) { return p10[v] - 1; }
//変換系
//[v] := iとなるようなvectorを返す
//存在しない物は-1
template<class T> auto keys(T& a) {    vector<decltype((a.begin())->fi)> res;    for (auto &&k :a)res.push_back(k.fi);    return res;}
template<class T> auto values(T& a) {    vector<decltype((a.begin())->se)> res;    for (auto &&k :a)res.push_back(k.se);    return res;}
template<class T, class U>  bool chma(T &a, const U &b) {    if (a < b) {        a = b;        return true;    }    return false;}
template<class T, class U>  bool chmi(T &a, const U &b) {    if (b < a) {        a = b;        return true;    }    return false;}
template<class T>  T min(T a, signed b) { return a < b ? a : b; }
template<class T>  T max(T a, signed b) { return a < b ? b : a; }
template<class T>  T min(T a, T b, T c) { return a >= b ? b >= c ? c : b : a >= c ? c : a; }
template<class T>  T max(T a, T b, T c) { return a <= b ? b <= c ? c : b : a <= c ? c : a; }
template<class T>  T min(vector<T>& a) { return *min_element(all(a)); }
template<class T>  T mini(vector<T>& a) { return min_element(all(a)) - a.begin(); }
template<class T>  T min(vector<T>& a, ll n) { return *min_element(a.begin(), a.begin() + min(n, sz(a))); }
template<class T>  T min(vector<T>& a, ll s, ll n) { return *min_element(a.begin() + s, a.begin() + min(n, sz(a))); }
template<class T>  T max(vector<T>& a) { return *max_element(all(a)); }
template<class T,class U>  T max(vector<T>& a,vector<U>& b) { return max(*max_element(all(a)),*max_element(all(b))); }
template<class T>  T maxi(vector<T>& a) { return max_element(all(a)) - a.begin(); }
template<class T>  T max(vector<T>& a, ll n) { return *max_element(a.begin(), a.begin() + min(n, sz(a))); }
template<class T>  T max(vector<T>& a, ll s, ll n) { return *max_element(a.begin() + s, a.begin() + min(n, sz(a))); }
template<typename A, size_t N>  A max(A (&a)[N]) {    A res = a[0];    rep(i, N)res = max(res, a[i]);    return res;}template<typename A, size_t N, size_t O>  A max(A (&a)[N][O]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P>  A max(A (&a)[N][O][P]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q>  A max(A (&a)[N][O][P][Q], const T &v) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R>  A max(A (&a)[N][O][P][Q][R]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S>  A max(A (&a)[N][O][P][Q][R][S]) {    A res = max(a[0]);    rep(i, N)res = max(res, max(a[i]));    return res;}
template<typename A, size_t N>  A min(A (&a)[N]) {    A res = a[0];    rep(i, N)res = min(res, a[i]);    return res;}template<typename A, size_t N, size_t O>  A min(A (&a)[N][O]) {    A res = min(a[0]);    rep(i, N)res = min(res, max(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P>  A min(A (&a)[N][O][P]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q>  A min(A (&a)[N][O][P][Q], const T &v) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R>  A min(A (&a)[N][O][P][Q][R]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}template<typename A, size_t N, size_t O, size_t P, size_t Q, size_t R, size_t S>  A min(A (&a)[N][O][P][Q][R][S]) {    A res = min(a[0]);    rep(i, N)res = min(res, min(a[i]));    return res;}
template<class T> T sum(vector<T> &v, ll s, ll t) {    T ret = 0;    rep(i, s, min(sz(v), t))ret += v[i];    return ret;}template<class T> T sum(vector<T> &v, ll t=inf) {    return sum(v, 0, t);}template<class T> T sum(vector<vector<T> > &v) {    T ret = 0;    rep(i, sz(v))ret += sum(v[i]);    return ret;}template<class T> T sum(vector<vector<vector<T> > > &v) {    T ret = 0;    rep(i, sz(v))ret += sum(v[i]);    return ret;}template<class T> T sum(vector<vector<vector<vector<T> > > > &v) {    T ret = 0;    rep(i, sz(v))ret += sum(v[i]);    return ret;}template<class T> T sum(vector<vector<vector<vector<vector<T> > > > > &v) {    T ret = 0;    rep(i, sz(v))ret += sum(v[i]);    return ret;}template<class T> auto sum(priority_queue<T, vector<T>, greater<T> > &r) {    auto q = r;    T ret = 0;    while (sz(q)) {        ret += q.top();        q.pop();    }    return ret;}template<class T> auto sum(priority_queue<T> &r) {    auto q = r;    T ret = 0;    while (sz(q)) {        ret += q.top();        q.pop();    }    return ret;}
//template<class T, class U, class... W>  auto sumn(vector<T> &v, U head, W... tail) {    auto ret = sum(v[0], tail...);    rep(i, 1, min(sz(v), head))ret += sum(v[i], tail...);    return ret;}
vi v_i(vi &a) {    int n = max(a) + 1;    vi ret(n, -1);    rep(i, sz(a)) {        ret[a[i]] = i;    }    return ret;}
void clear(PQ &q) { q = PQ(); }
void clear(priority_queue<int>&q) { q = priority_queue<int>();}
template<class T> void clear(queue<T> &q) { while (q.size())q.pop(); }
template<class T> T *negarr(ll size) {    T *body = (T *) malloc((size * 2 + 1) * sizeof(T));    return body + size;}
template<class T> T *negarr2(ll h, ll w) {    double **dummy1 = new double *[2 * h + 1];    double *dummy2 = new double[(2 * h + 1) * (2 * w + 1)];    dummy1[0] = dummy2 + w;    for (ll i = 1; i <= 2 * h + 1; ++i) { dummy1[i] = dummy1[i - 1] + 2 * w + 1; }    double **a = dummy1 + h;    return a;}
//imoは0-indexed
//ruiは1-indexed
template<class T> vector<T> imo(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)ret[i + 1] += ret[i];    return ret;}
//kと同じものの数
template<class T, class U> vi imo(vector<T> &a, U k) {vector<T> ret = a;rep(i, sz(ret))ret[i] = a[i] == k;rep(i, sz(ret) - 1)ret[i + 1] += ret[i];return ret;}
template<class T> vector<T> imox(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)ret[i + 1] ^= ret[i];    return ret;}
//漸化的に最小を持つ
template<class T> vector<T> imi(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)chmi(ret[i + 1], ret[i]);    return ret;}
template<class T> vector<T> ima(vector<T> &v) {    vector<T> ret = v;    rep(i, sz(ret) - 1)chma(ret[i + 1], ret[i]);    return ret;}
template<class T> vector<T> rimi(vector<T> &v) {    vector<T> ret = v;    rer(i, sz(ret) - 1, 1)chmi(ret[i - 1], ret[i]);    return ret;}
template<class T> vector<T> rima(vector<T> &v) {    vector<T> ret = v;    rer(i, sz(ret) - 1, 1)chma(ret[i - 1], ret[i]);    return ret;}
template<class T> struct ruiC {
    vector<T> rui;
    ruiC(vector<T> &ru) : rui(ru) {}
    /*先頭0*/
    ruiC() : rui(1, 0) {}
    T operator()(ll l, ll r) {        if (l > r) {            cerr << "ruic ";            deb(l, r);            assert(0);        }        return rui[r] - rui[l];    }
    T operator()(int r) { return operator()(0, r); }
    /*単体iを返す 累積でないことに注意(seg木との統一でこうしている)*/
    T operator[](ll i) { return rui[i + 1] - rui[i]; }
    /*0から順に追加される必要がある*/
    void operator+=(T v) { rui.push_back(rui.back() + v); }
    void add(int i, T v) {if (sz(rui) - 1 != i)ole();operator+=(v);}
    T back() { return rui.back(); }
    ll size() { return rui.size(); }
};
template<class T> struct rruic {
    const T *rrui;
    rruic(T *ru) : rrui(ru) {}
    //[r , l)  :  [3 , -1)等
    T operator()(ll r, ll l) {assert(r >= l);return rrui[l] - rrui[r];}
    T operator()(int l){ return rrui[l];}
    T operator[](ll i) { return rrui[i-1] - rrui[i]; }
};
template<class T>ostream &operator<<(ostream &os, ruiC<T> a) {fora(v,a.rui)os<<v<<" ";return os;}
template<class T> vector<T> ruiv(vector<T> &a) {    vector<T> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + a[i];    return ret;}
template<class T> ruiC<T> ruic() {return ruiC<T>();}
template<class T> ruiC<T> ruic(vector<T> &a) {    vector<T> ret = ruiv(a);    return ruiC<T>(ret);}
vector<ll> ruiv(string &a) {    if (sz(a) == 0)return vi(1);    ll dec = ('0' <= a[0] && a[0] <= '9') ? '0' : 0;    vector<ll> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + a[i] - dec;    return ret;}
ruiC<ll> ruic(string &a) {    vector<ll> ret = ruiv(a);    return ruiC<ll>(ret);}
//kと同じものの数
template<class T, class U> vi ruiv(T &a, U k) {    vi ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] + (a[i] == k);    return ret;}
template<class T, class U> ruiC<ll> ruic(T &a, U k) {    vi ret = ruiv(a, k);    return ruiC<ll>(ret);}

//h query
template<class T> vector<T> imoh(vector<vector<T>> &v, int w) {vector<T> ret(sz(v));rep(h,sz(ret)){ret[h] = v[h][w];}rep(i, sz(ret) - 1){ret[i + 1] += ret[i];}return ret;}
template<class T> vector<T> ruih(vector<vector<T>> &v, int w) {vector<T> ret(sz(v) + 1);rep(h, sz(v)) { ret[h + 1] = v[h][w]; }rep(i, sz(v)) { ret[i + 1] += ret[i]; }return ret;}
template<class T> ruiC<T> ruihc(vector<vector<T>> &a, int w) {vector<T> ret = ruih(a, w);return ruiC<T>(ret);}

//xor
template<class T> struct ruixC {    const vector<T> rui;    ruixC(vector<T> &ru) : rui(ru) {}    T operator()(ll l, ll r) {        if (l > r) {            cerr << "ruiXc ";            deb(l, r);            assert(0);        }        return rui[r] ^ rui[l];    }    T operator[](ll i) { return rui[i]; }    T back() { return rui.back(); }    ll size() { return rui.size(); }};
template<class T> vector<T> ruix(vector<T> &a) {    vector<T> ret(a.size() + 1);    rep(i, a.size())ret[i + 1] = ret[i] ^ a[i];    return ret;}
template<class T> ruixC<ll> ruixc(vector<T>  &a) {vi ret = ruix(a);return ruixC<ll>(ret);}
template<class T> vector<T> ruim(vector<T> &a) {    vector<T> res(a.size() + 1, 1);    rep(i, a.size())res[i + 1] = res[i] * a[i];    return res;}
//漸化的に最小を1indexで持つ
template<class T> vector<T> ruimi(vector<T> &a) {    ll n = sz(a);    vector<T> ret(n + 1);    rep(i, 1, n) {        ret[i] = a[i - 1];        chmi(ret[i + 1], ret[i]);    }    return ret;}
//template<class T> T *rrui(vector<T> &a) {
//右から左にかけての半開区間 (-1 n-1]
template<class T> rruic<T> rrui(vector<T> &a) {    ll len = a.size();    T *body = (T *) malloc((len + 1) * sizeof(T));    T *res = body + 1;    rer(i, len - 1)res[i - 1] = res[i] + a[i];    return rruic<T>(res);}
//掛け算
template<class T> T *rruim(vector<T> &a) {    ll len = a.size();    T *body = (T *) malloc((len + 1) * sizeof(T));    T *res = body + 1;    res[len - 1] = 1;    rer(i, len - 1)res[i - 1] = res[i] * a[i];    return res;}
template<class T, class U> void inc(T &a, U v = 1) { a += v; }
template<class T, class U> void inc(vector<T> &a, U v = 1) { for (auto &u:a)inc(u, v); }
template<class T, class U> void dec(T &a, U v = 1) { a -= v; }
template<class T, class U> void dec(vector<T> &a, U v = 1) { for (auto &u :a)dec(u, v); }
template<class U> void dec(string &a, U v = 1) { for (auto &u :a)dec(u, v); }
template<class T> void dec(vector<T> &a) { for (auto &u :a)dec(u, 1); }
template<class T,class U> void dec(vector<T> &a,vector<U> &b) { for (auto &u :a)dec(u, 1);for (auto &u :b)dec(u, 1); }
template<class T,class U,class W> void dec(vector<T> &a,vector<U> &b,vector<W>&c ) { for (auto &u :a)dec(u, 1);for (auto &u :b)dec(u, 1);for (auto &u :c)dec(u, 1); }
bool ins(ll h, ll w, ll H, ll W) { return h >= 0 && w >= 0 && h < H && w < W; }
bool ins(ll l, ll v, ll r) { return l <= v && v < r; }
template<class T> bool ins(vector<T> &a, ll i, ll j = 0) { return ins(0, i, sz(a)) && ins(0, j, sz(a)); }
ll u(ll a) { return a < 0 ? 0 : a; }
template<class T> vector<T> u(const vector<T> &a) {    vector<T> ret = a;    fora(v, ret)v = u(v);    return ret;}
#define MIN(a) numeric_limits<a>::min()
#define MAX(a) numeric_limits<a>::max()

//添え字を返す
template<class F> ll goldd_l(ll left, ll right, F calc) {    double GRATIO = 1.6180339887498948482045868343656;    ll lm = left + (ll) ((right - left) / (GRATIO + 1.0));    ll rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));    ll fl = calc(lm);    ll fr = calc(rm);    while (right - left > 10) {        if (fl < fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + (ll) ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        } else {            left = lm;            lm = rm;            fl = fr;            rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    ll minScore = MAX(ll);    ll resIndex = left;    for (ll i = left; i < right + 1; ++i) {        ll score = calc(i);        if (minScore > score) {            minScore = score;            resIndex = i;        }    }    return resIndex;}
template<class F> ll goldt_l(ll left, ll right, F calc) {        double GRATIO = 1.6180339887498948482045868343656;        ll lm = left + (ll) ((right - left) / (GRATIO + 1.0));        ll rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));        ll fl = calc(lm);        ll fr = calc(rm);        while (right - left > 10) {            if (fl > fr) {                right = rm;                rm = lm;                fr = fl;                lm = left + (ll) ((right - left) / (GRATIO + 1.0));                fl = calc(lm);            } else {                left = lm;                lm = rm;                fl = fr;                rm = lm + (ll) ((right - lm) / (GRATIO + 1.0));                fr = calc(rm);            }        }    if (left > right) {        ll l = left;        left = right;        right = l;    }    ll maxScore = MIN(ll);    ll resIndex = left;    for (ll i = left; i < right + 1; ++i) {        ll score = calc(i);        if (maxScore < score) {            maxScore = score;            resIndex = i;        }    }    return resIndex;}
/*loopは200にすればおそらく大丈夫 余裕なら300に*/
template<class F> dou goldd_d(dou left, dou right, F calc, ll loop = 200) {    dou GRATIO = 1.6180339887498948482045868343656;    dou lm = left + ((right - left) / (GRATIO + 1.0));    dou rm = lm + ((right - lm) / (GRATIO + 1.0));    dou fl = calc(lm);    dou fr = calc(rm);    /*200にすればおそらく大丈夫*/    /*余裕なら300に*/    ll k = 141;    loop++;    while (--loop) {        if (fl < fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        } else {            left = lm;            lm = rm;            fl = fr;            rm = lm + ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    return left;}
template<class F> dou goldt_d(dou left, dou right, F calc, ll loop = 200) {    double GRATIO = 1.6180339887498948482045868343656;    dou lm = left + ((right - left) / (GRATIO + 1.0));    dou rm = lm + ((right - lm) / (GRATIO + 1.0));    dou fl = calc(lm);    dou fr = calc(rm);    loop++;    while (--loop) {        if (fl > fr) {            right = rm;            rm = lm;            fr = fl;            lm = left + ((right - left) / (GRATIO + 1.0));            fl = calc(lm);        } else {            left = lm;            lm = rm;            fl = fr;            rm = lm + ((right - lm) / (GRATIO + 1.0));            fr = calc(rm);        }    }    return left;}
//l ~ rを複数の区間に分割し、極致を与えるiを返す time-20 msまで探索
template<class F> ll goldd_ls(ll l, ll r, F calc, ll time = 2000) {    auto lim = milliseconds(time - 20);    ll mini = 0, minv = MAX(ll);    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        ll haba = (r - l + k) / k;/*((r-l+1) + k-1) /k*/        ll nl = l;        ll nr = l + haba;        rep(i, k) {            ll ni = goldd_l(nl, nr, calc);            if (chmi(minv, calc(ni))) mini = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return mini;}
template<class F> ll goldt_ls(ll l, ll r, F calc, ll time = 2000) {    auto lim = milliseconds(time - 20);    ll maxi = 0, maxv = MIN(ll);    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        ll haba = (r - l + k) / k;/*((r-l+1) + k-1) /k*/        ll nl = l;        ll nr = l + haba;        rep(i, k) {            ll ni = goldt_l(nl, nr, calc);            if (chma(maxv, calc(ni))) maxi = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return maxi;}
template<class F> dou goldd_d_s(dou l, dou r, F calc, ll time = 2000) {    /*20ms余裕を持つ*/    auto lim = milliseconds(time - 20);    dou mini = 0, minv = MAX(dou);    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        dou haba = (r - l) / k;        dou nl = l;        dou nr = l + haba;        rep(i, k) {            dou ni = goldd_d(nl, nr, calc);            if (chmi(minv, calc(ni))) mini = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return mini;}
template<class F> dou goldt_d_s(dou l, dou r, F calc, ll time = 2000) {    /*20ms余裕を残している*/    auto lim = milliseconds(time - 20);    dou maxi = 0, maxv = MIN(dou);    /*区間をk分割する*/    rep(k, 1, inf) {        auto s = system_clock::now();        dou haba = (r - l) / k;        dou nl = l;        dou nr = l + haba;        rep(i, k) {            dou ni = goldt_d(nl, nr, calc);            if (chma(maxv, calc(ni))) maxi = ni;            nl = nr;            nr = nl + haba;        }        auto end = system_clock::now();        auto part = duration_cast<milliseconds>(end - s);        auto elapsed = duration_cast<milliseconds>(end - start_time);        if (elapsed + part * 2 >= lim) { break; }    }    return maxi;}
template<class T> T min(vector<vector<T >> &a) {    T res = MAX(T);    rep(i, a.size())chmi(res, *min_element(all(a[i])));    return res;}
template<class T> T max(vector<vector<T >> &a) {    T res = MIN(T);    rep(i, a.size())chma(res, *max_element(all(a[i])));    return res;}
constexpr bool bget(ll m, ll keta) {
    #ifdef _DEBUG
        assert(keta<=62);//オーバーフロー 1^62までしか扱えない
    #endif
    return (m >> keta) & 1; }
ll bget(ll m, ll keta, ll sinsuu) {    m /= (ll) pow(sinsuu, keta);    return m % sinsuu;}
constexpr ll bit(ll n) {
    #ifdef _DEBUG
        assert(n<=62);//オーバーフロー 1^62までしか扱えない
    #endif
    return (1LL << (n)); }
ll bit(ll n, ll sinsuu) { return (ll) pow(sinsuu, n); }
ll mask(ll n) { return (1ll << n) - 1; }
//aをbitに置きなおす
ll bit(vi&a){    int m=0;    for(auto&& v:a) m|=bit(v);    return m;}
#define bcou __builtin_popcountll
//最下位ビット
ll lbit(ll n) { return n & -n; }
ll lbiti(ll n) { return log2(n & -n); }
//最上位ビット
ll hbit(ll n) {    n |= (n >> 1);    n |= (n >> 2);    n |= (n >> 4);    n |= (n >> 8);    n |= (n >> 16);    n |= (n >> 32);    return n - (n >> 1);}
ll hbiti(ll n){return log2(hbit(n));}
ll hbitk(ll n) {    ll k = 0;    rer(i, 5) {        ll a = k + (1ll << i);        ll b = 1ll << a;        if (b <= n)k += 1ll << i;    }    return k;}
//初期化は0を渡す
ll nextComb(ll &mask, ll n, ll r) {    if (!mask)return mask = (1LL << r) - 1;    ll x = mask & -mask; /*最下位の1*/    ll y = mask + x; /*連続した下の1を繰り上がらせる*/    ll res = ((mask & ~y) / x >> 1) | y;    if (bget(res, n))return mask = 0; else return mask = res;}
//n桁以下でビットがr個立っているもののvectorを返す
vi bitCombList(ll n, ll r) {    vi res;    ll m = 0;    while (nextComb(m, n, r)) { res.push_back(m); }    return res;}
//masの立ってるindexを見る
#define forbit(i, mas) for (int forbitj = lbit(mas), forbitm = mas, i = log2(forbitj); forbitm; forbitm = forbitj ? forbitm ^ forbitj : 0, forbitj = lbit(forbitm), i = log2(forbitj))

char itoal(ll i) { return 'a' + i; }
char itoaL(ll i) { return 'A' + i; }
ll altoi(char c) {    if ('A' <= c && c <= 'Z')return c - 'A';    return c - 'a';}
ll ctoi(char c) { return c - '0'; }
char itoc(ll i) { return i + '0'; }
ll vtoi(vi &v) {    ll res = 0;    if (sz(v) > 18) {        debugline("vtoi");        deb(sz(v));        ole();    }    rep(i, sz(v)) {        res *= 10;        res += v[i];    }    return res;}
vi itov(ll i) {    vi res;    while (i) {        res.push_back(i % 10);        i /= 10;    }    rev(res);    return res;}
vi stov(string &a) {    ll n = sz(a);    vi ret(n);    rep(i, n) { ret[i] = a[i] - '0'; }    return ret;}
//基準を満たさないものは0になる
vi stov(string &a, char one) {    ll n = sz(a);    vi ret(n);    rep(i, n)ret[i] = a[i] == one;    return ret;}
vector<vector<ll>> ctoi(vector<vector<char>> s, char c) {    ll n = sz(s), m = sz(s[0]);    vector<vector<ll>> res(n, vector<ll>(m));    rep(i, n)rep(j, m)res[i][j] = s[i][j] == c;    return res;}
#define unique(v) v.erase( unique(v.begin(), v.end()), v.end() );
//[i] := vを返す
//aは0~n-1で置き換えられる
vi compress(vi &a) {    vi b;    ll len = a.size();    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) { a[i] = lower_bound(all(b), a[i]) - b.begin(); }    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vi &a, umap<ll, ll> &map) {    vi b;    ll len = a.size();    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) {        ll v = a[i];        a[i] = lower_bound(all(b), a[i]) - b.begin();        map[v] = a[i];    }    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vi &a, vi &r) {    vi b;    ll len = a.size();    fora(v, a)b.push_back(v);    fora(v, r)b.push_back(v);    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) a[i] = lower_bound(all(b), a[i]) - b.begin();    for (ll i = 0; i < sz(r); ++i) r[i] = lower_bound(all(b), r[i]) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vi &a, vi &r, vi &s) {    vi b;    ll len = a.size();    fora(v, a)b.push_back(v);    fora(v, r)b.push_back(v);    fora(v, s)b.push_back(v);    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) a[i] = lower_bound(all(b), a[i]) - b.begin();    for (ll i = 0; i < sz(r); ++i) r[i] = lower_bound(all(b), r[i]) - b.begin();    for (ll i = 0; i < sz(s); ++i) r[i] = lower_bound(all(b), s[i]) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vector<vi> &a) {    vi b;    fora(vv, a)fora(v, vv)b.push_back(v);    sort(b);    unique(b);    fora(vv, a)fora(v, vv)v = lower_bound(all(b), v) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
vi compress(vector<vector<vi >> &a) {    vi b;    fora(vvv, a)fora(vv, vvv)fora(v, vv)b.push_back(v);    sort(b);    unique(b);    fora(vvv, a)fora(vv, vvv)fora(v, vv)v = lower_bound(all(b), v) - b.begin();    ll blen = sz(b);    vi ret(blen);    rep(i, blen) { ret[i] = b[i]; }    return ret;}
void compress(ll a[], ll len) {    vi b;    for (ll i = 0; i < len; ++i) { b.push_back(a[i]); }    sort(b);    unique(b);    for (ll i = 0; i < len; ++i) { a[i] = lower_bound(all(b), a[i]) - b.begin(); }}
//要素が見つからなかったときに困る
#define binarySearch(a, v) (binary_search(all(a),v))
#define lowerIndex(a, v) (lower_bound(all(a),v)-a.begin())
#define lowerBound(a, v) (*lower_bound(all(a),v))
#define upperIndex(a, v) (upper_bound(all(a),v)-a.begin())
#define upperBound(a, v) (*upper_bound(all(a),v))
#define rlowerIndex(a, v) (upper_bound(all(a),v)-a.begin()-1)
#define rlowerBound(a, v) *(--(upper_bound(all(a),v)))
#define rupperIndex(a, v) (lower_bound(all(a),v)-a.begin()-1)
#define rupperBound(a, v) *(--(lower_bound(all(a),v)))
#define next2(a) next(next(a))
#define prev2(a) prev(prev(a))

//狭義の単調増加列 長さを返す
template<class T> int lis(vector<T> &a) {    int n = sz(a);    vi tail(n + 1, MAX(T));    rep(i, n) {        int id = lowerIndex(tail, a[i]);/**/        tail[id] = a[i];    }    return lowerIndex(tail, MAX(T));}
template<class T> int lis_eq(vector<T> &a) {    int n = sz(a);    vi tail(n + 1, MAX(T));    rep(i, n) {        int id = upperIndex(tail, a[i]);/**/        tail[id] = a[i];    }    return lowerIndex(tail, MAX(T));}

//iteratorを返す
//valueが1以上の物を返す 0は見つけ次第削除
//vを減らす場合 (*it).se--でいい
template<class T, class U, class V> auto lower_map(map<T, U> &m, V k) {    auto ret = m.lower_bound(k);    while (ret != m.end() && (*ret).second == 0) {        ret = m.erase(ret);    }    return ret;}
template<class T, class U, class V> auto upper_map(map<T, U> &m, V k) {    auto ret = m.upper_bound(k);    while (ret != m.end() && (*ret).second == 0) {        ret = m.erase(ret);    }    return ret;}
//存在しなければエラー
template<class T, class U, class V> auto rlower_map(map<T, U> &m, V k) {    auto ret = upper_map(m, k);    assert(ret != m.begin());    ret--;    while (1) {        if ((*ret).second != 0)break;        assert(ret != m.begin());        auto next = ret;        --next;        m.erase(ret);        ret = next;    }    return ret;}
template<class T, class U, class V> auto rupper_map(map<T, U> &m, V k) {    auto ret = lower_map(m, k);    assert(ret != m.begin());    ret--;    while (1) {        if ((*ret).second != 0)break;        assert(ret != m.begin());        auto next = ret;        --next;        m.erase(ret);        ret = next;    }    return ret;}

template<class T> void fin(T s) { cout << s << endl, exit(0); }

//便利 数学 math
//sub ⊂ top
bool subset(int sub, int top) {return (sub & top) == sub;}
//-180 ~ 180 degree
double atand(double h, double w) {return atan2(h, w) / PI * 180;}
ll mod(ll a, ll m) { return (a % m + m) % m; }
ll pow(ll a) { return a * a; };
ll fact(ll v) { return v <= 1 ? 1 : v * fact(v - 1); }
dou factd(int v){static vd fact(2,1);    if(sz(fact)<=v){        rep(i,sz(fact),v+1){            fact.push_back(fact.back()*i);        }    }    return fact[v];}

ll comi(ll n, ll r) {    assert(n < 100);    static vvi(pas, 100, 100);    if (pas[0][0])return pas[n][r];    pas[0][0] = 1;    rep(i, 1, 100) {        pas[i][0] = 1;        rep(j, 1, i + 1)pas[i][j] = pas[i - 1][j - 1] + pas[i - 1][j];    }    return pas[n][r];}
double comd2(ll n, ll r) {    static vvd(comb, 2020, 2020);    if (comb[0][0] == 0) {        comb[0][0] = 1;        rep(i, 2000) {            comb[i + 1][0] = 1;            rep(j, 1, i + 2) { comb[i + 1][j] = comb[i][j] + comb[i][j - 1]; }        }    }    return comb[n][r];}
double comd(int n, int r) {    if (r < 0 || r > n) return 0;    if (n < 2020)return comd2(n, r);    static vd fact(2, 1);    if (sz(fact) <= n) { rep(i, sz(fact), n + 1) { fact.push_back(fact.back() * i); }}    return fact[n] / fact[n - r] / fact[r];}

ll gcd(ll a, ll b) {while (b) a %= b, swap(a, b);return abs(a);}
ll gcd(vi b) {ll res = b[0];rep(i, 1, sz(b))res = gcd(b[i], res);return res;}
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
ll lcm(vi a) {ll res = a[0];rep(i, 1, sz(a))res = lcm(a[i], res);return res;}
ll ceil(ll a, ll b) {if (b == 0) {debugline("ceil");deb(a, b);ole();return -1;} else if (a < 0) { return 0; } else { return (a + b - 1) / b; }}

// ax + r (x は非負整数) で表せる整数のうち、v 以上となる最小の整数
ll lower_rem__kr(ll a, ll r, ll v) {if (r >= v) return r;return (v - r + a-1) / a * a + r;}
//第何項か
ll lower_rem_i__kr(ll a, ll r, ll v) {if (r >= v) return 0;return (v - r + a-1) / a;}
ll upper_rem__kr(ll a, ll r, ll v) {return lower_rem__kr(a,r,v+1);}
ll upper_rem_i__kr(ll a, ll r, ll v) {    return lower_rem_i__kr(a,r,v+1);}


//v * v >= aとなる最小のvを返す
ll sqrt(ll a) {if (a < 0) {debugline("sqrt");deb(a);ole();}ll res = (ll) std::sqrt(a);while (res * res < a)++res;return res;}
double log(double e, double x) { return log(x) / log(e); }
ll sig(ll t) { return ((1 + t) * t) >> 1; }
ll sig(ll s, ll t) { return ((s + t) * (t - s + 1)) >> 1; }


//機能拡張
constexpr int bsetlen = k5*2;
#define bset bitset<bsetlen>
bool operator<(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] < b[i])return true;if (a[i] > b[i])return false;}return false;}
bool operator>(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] > b[i])return true;if (a[i] < b[i])return false;}return false;}
bool operator<=(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] < b[i])return true;if (a[i] > b[i])return false;}return true;}
bool operator>=(bitset<bsetlen> &a, bitset<bsetlen> &b) {rer(i, bsetlen - 1) {if (a[i] > b[i])return true;if (a[i] < b[i])return false;}return true;}
string operator~(string &a) {string res = a;for (auto &&c:res) {if (c == '0')c = '1';else if (c == '1')c = '0';else {cerr << "cant ~" << a << "must bit" << endl;exit(0);}}return res;}

template<class T> void seth(vector<vector<T>> &S, int w, vector<T> &v) {assert(sz(S) == sz(v));assert(w < sz(S[0]));rep(h, sz(S)) { S[h][w] = v[h]; }}

template<class T, class U> void operator+=(pair<T,U> &a, pair<T,U> & b) {a.fi+=b.fi;a.se+=b.se;}
template<class T, class U> pair<T,U> operator+(pair<T,U> &a, pair<T,U> & b) {return pair<T,U>(a.fi+b.fi,a.se+b.se);}

template<typename CharT, typename Traits, typename Alloc>basic_string<CharT, Traits, Alloc>operator+(const basic_string<CharT, Traits, Alloc> &lhs, const int rv) {    basic_string<CharT, Traits, Alloc> str(lhs);    str.append(to_string(rv));    return str;}template<typename CharT, typename Traits, typename Alloc>void operator+=(basic_string<CharT, Traits, Alloc> &lhs, const int rv) {    lhs += to_string(rv);}template<typename CharT, typename Traits, typename Alloc>basic_string<CharT, Traits, Alloc>operator+(const basic_string<CharT, Traits, Alloc> &lhs, const signed rv) {    basic_string<CharT, Traits, Alloc> str(lhs);    str.append(to_string(rv));    return str;}template<typename CharT, typename Traits, typename Alloc>void operator+=(basic_string<CharT, Traits, Alloc> &lhs, const signed rv) {    lhs += to_string(rv);}
template<typename CharT, typename Traits, typename Alloc> void operator*=(basic_string<CharT, Traits, Alloc> &s, int num) {auto bek = s;s = "";for (; num; num >>= 1) {if (num & 1) {s += bek;}bek += bek;}}
template<class T, class U> void operator+=(queue<T> &a, U v) { a.push(v); }template<class T, class U> void operator+=(deque<T> &a, U v) { a.push_back(v); }template<class T> priority_queue<T, vector<T>, greater<T> > &operator+=(priority_queue<T, vector<T>, greater<T> > &a, vector <T> &v) {    fora(d, v)a.push(d);    return a;}template<class T, class U> priority_queue<T, vector<T>, greater<T> > &operator+=(priority_queue<T, vector<T>, greater<T> > &a, U v) {    a.push(v);    return a;}template<class T, class U> priority_queue<T> &operator+=(priority_queue<T> &a, U v) {    a.push(v);    return a;}template<class T> set<T> &operator+=(set<T> &a, vector <T> v) {    fora(d, v)a.insert(d);    return a;}template<class T, class U> auto operator+=(set<T> &a, U v) { return a.insert(v); }template<class T, class U> auto operator-=(set<T> &a, U v) { return a.erase(v); }template<class T, class U> auto operator+=(mset<T> &a, U v) { return a.insert(v); }template<class T, class U> set<T, greater<T>> &operator+=(set<T, greater<T>> &a, U v) {    a.insert(v);    return a;}template<class T, class U> vector<T> &operator+=(vector<T> &a, U v) {    a.push_back(v);    return a;}template<class T, class U> vector<T> operator+(const vector <T> &a, U v) {    vector<T> ret = a;    ret += v;    return ret;}template<class T, class U> vector<T> operator+(U v, const vector <T> &a) {    vector<T> ret = a;    ret.insert(ret.begin(), v);    return ret;}template<class T> vector<T> operator+(vector<T> a, vector <T> b) {    vector<T> ret;    ret = a;    fora(v, b)ret += v;    return ret;}template<class T> vector<T> &operator+=(vector<T> &a, vector <T> &b) {    rep(i, sz(b)) {/*こうしないとa+=aで両辺が増え続けてバグる*/        a.push_back(b[i]); }    return a;}template<class T, class U> map<T, U> &operator+=(map<T, U> &a, map<T,U> &b) {fora(bv,b){        a[bv.first] += bv.second;    }    return a;}
template<class T> vector<T> &operator-=(vector<T> &a, vector <T> &b) {    if (sz(a) != sz(b)) {        debugline("vector<T> operator-=");        deb(a);        deb(b);        exit(0);    }    rep(i, sz(a))a[i] -= b[i];    return a;}
template<class T> vector<T> operator-(vector<T> &a, vector<T> &b) {    if (sz(a) != sz(b)) {        debugline("vector<T> operator-");        deb(a);        deb(b);        ole();    }    vector<T> res(sz(a));    rep(i, sz(a))res[i] = a[i] - b[i];    return res;}
template<class T, class U> void operator*=(vector<T> &a, U b) {    vector<T> ta = a;    rep(b-1){        a+=ta;    }}
template<typename T> void erase(vector<T> &v, unsigned ll i) { v.erase(v.begin() + i); }
template<typename T> void erase(vector<T> &v, unsigned ll s, unsigned ll e) { v.erase(v.begin() + s, v.begin() + e); }
template<class T, class U> void erase(map<T, U> &m, ll okl, ll ngr) { m.erase(m.lower_bound(okl), m.lower_bound(ngr)); }
template<class T> void erase(set<T> &m, ll okl, ll ngr) { m.erase(m.lower_bound(okl), m.lower_bound(ngr)); }
template<typename T> void erasen(vector<T> &v, unsigned ll s, unsigned ll n) { v.erase(v.begin() + s, v.begin() + s + n); }
template<typename T, typename U> void insert(vector<T> &v, unsigned ll i, U t) { v.insert(v.begin() + i, t); }
template<typename T, typename U> void push_front(vector<T> &v, U t) { v.insert(v.begin(), t); }
template<typename T, typename U> void insert(vector<T> &v, unsigned ll i, vector<T> list) { for (auto &&va:list)v.insert(v.begin() + i++, va); }
template<typename T> void insert(set<T> &v, vector<T> list) { for (auto &&va :list)v.insert(va); }
vector<string> split(const string a, const char deli) {    string b = a + deli;    ll l = 0, r = 0, n = b.size();    vector<string> res;    rep(i, n) {        if (b[i] == deli) {            r = i;            if (l < r)res.push_back(b.substr(l, r - l));            l = i + 1;        }    }    return res;}
vector<string> split(const string a, const string deli) {    vector<string> res;    ll kn = sz(deli);    std::string::size_type Pos(a.find(deli));    ll l = 0;    while (Pos != std::string::npos) {        if (Pos - l)res.push_back(a.substr(l, Pos - l));        l = Pos + kn;        Pos = a.find(deli, Pos + kn);    }    if (sz(a) - l)res.push_back(a.substr(l, sz(a) - l));    return res;}
void yn(bool a) { if (a)cout << "yes" << endl; else cout << "no" << endl; }
void Yn(bool a) { if (a)cout << "Yes" << endl; else cout << "No" << endl; }
void YN(bool a) { if (a)cout << "YES" << endl; else cout << "NO" << endl; }
void fyn(bool a) {    if (a)cout << "yes" << endl; else cout << "no" << endl;    exit(0);}
void fYn(bool a) {    if (a)cout << "Yes" << endl; else cout << "No" << endl;    exit(0);}
void fYN(bool a) {    if (a)cout << "YES" << endl; else cout << "NO" << endl;    exit(0);}
void Possible(bool a) {    if (a)cout << "Possible" << endl; else cout << "Impossible" << endl;    exit(0);}
void POSSIBLE(bool a) {    if (a)cout << "POSSIBLE" << endl; else cout << "IMPOSSIBLE" << endl;    exit(0);}
template<typename T> class fixed_point        : T {public:    explicit constexpr fixed_point(T &&t) noexcept: T(std::forward<T>(t)) {}    template<typename... Args> constexpr decltype(auto) operator()(Args &&... args) const { return T::operator()(*this, std::forward<Args>(args)...); }};template<typename T> static inline constexpr decltype(auto) fix(T &&t) noexcept { return fixed_point<T>{std::forward<T>(t)}; }


//@起動時
struct initon {
    initon() {
        cin.tie(0);
        ios::sync_with_stdio(false);
        cout.setf(ios::fixed);
        cout.precision(16);
        srand((unsigned) clock() + (unsigned) time(NULL));
    };
} initonv;//@formatter:on

//gra mll pr
//上下左右
const string udlr = "udlr";
string UDLR = "UDLR";//x4と連動 UDLR.find('U') := x4[0]
//右、上が正
constexpr ll y4[] = {1, -1, 0, 0};
constexpr ll x4[] = {0, 0, -1, 1};
constexpr ll y8[] = {0, 1, 0, -1, -1, 1, 1, -1};
constexpr ll x8[] = {1, 0, -1, 0, 1, -1, 1, -1};
//ll k, K, d, x, y, z, q, Q,cou;vvi (s, 0, 0);

//ll n, m, k, H, W;
//vi a, b;
//vvc (ba, 0, 0);

ll N, M, H, W;
vi A, B, C;
/*@formatter:off*/
#define forg(gi, ve) for (ll gi = 0,forglim = ve.size(), f, t, c; gi < forglim && (f = ve[gi].f, t = ve[gi].t, c = ve[gi].c, true); ++gi)
#define fort(gi, ve) for (ll gi = 0, f, t, c; gi < ve.size() && (f = ve[gi].f, t = ve[gi].t, c = ve[gi].c, true); ++gi)if(t!=p)
#define fore(gi, ve) for (ll gi = 0,forglim = ve.size(), f, t, c; gi < forglim && (f = ve[gi].f, t = ve[gi].t, c = ve[gi].c, true); ++gi)

template<class T> struct edge {    int f, t;    T c;    edge(int f, int t, T c = 1) : f(f), t(t), c(c) {}    bool operator<(const edge &b) const { return c < b.c; }    bool operator>(const edge &b) const { return c > b.c; }};
template<class T> ostream &operator<<(ostream &os, edge<T> &e) {os << e.f << " " << e.t << " " << e.c;return os;}
template<typename T> class graph {
public :
    vector<vector<edge<T>>> g;
    vector<edge<T>> edges;
    int n;
    explicit graph(int n) : n(n) { g.resize(n); }
    void clear() { g.clear(), edges.clear(); }
    void resize(int n) {this->n = n;g.resize(n);}
    int size() { return n; }
    vector<edge<T> > &operator[](int i) { return g[i]; }
    virtual void add(int f, int t, T c) = 0;
    virtual void set_edges() = 0;
};
template<typename T =ll> class digraph : public graph<T> {
public:
    using graph<T>::g;
    using graph<T>::n;
    using graph<T>::edges;
    explicit digraph(int n) : graph<T>(n) {}
    void add(int f, int t, T c = 1) {
        if (!(0 <= f && f < n && 0 <= t && t < n)) {debugline("digraph add");deb(f, t, c);ole();}
        g[f].emplace_back(f, t, c);
        edges.emplace_back(f, t, c);//edgesを使わないなら消せる
    }
    void ing(int n, int m, int minus = 1) {        this->resize(n);        rep(i, m) {            int f, t;            cin >> f >> t;            f -= minus;            t -= minus;            add(f, t);        }    }
    void ingc(int n, int m, int minus = 1) {        this->resize(n);        rep(i, m) {            int f, t, c;            cin >> f >> t >> c;            f -= minus;            t -= minus;            add(f, t, c);        }    }
    void set_edges() override{        if (sz(edges))return;        rep(i, n)fora(e, g[i])edges.push_back(e);    }
};
template<class T=int> class undigraph : public graph<T> {
public:
    using graph<T>::g;    using graph<T>::n;    using graph<T>::edges;
    explicit undigraph(int n) : graph<T>(n) {}
    // f < t
    void add(int f, int t, T c = 1) {
        if (!(0 <= f && f < n && 0 <= t && t < n)) {            debugline("undigraph add");            deb(f, t, c);            ole();        }
        g[f].emplace_back(f, t, c);
        g[t].emplace_back(t, f, c);
        edges.emplace_back(f, t, c);//edgesを使わないなら消せる
        edges.emplace_back(t, f, c);
    }
    void add(edge<T> &e) {        int f = e.f, t = e.t;        T c = e.c;        add(f, t, c);    }
    void ing(int n, int m, int minus = 1) {        this->resize(n);        rep(i, m) {            int f, t;            cin >> f >> t;            f -= minus;            t -= minus;            add(f, t);        }    }
    void ingc(int n, int m, int minus = 1) {        this->resize(n);        rep(i, m) {            int f, t, c;            cin >> f >> t >> c;            f -= minus;            t -= minus;            add(f, t, c);        }    }
    void set_edges () override{        if (sz(edges))return;        rep(i, n)fora(e, g[i])edges.push_back(e);    }
};
template<class T> vi dis_path(digraph<T> &g, vector<T> &dis, int s, int t, int init_value) {    assert(dis[t] != init_value);    auto rg = rev(g);    int now = t;    vi path;    path.push_back(now);    T cost = 0;    while (now != s) {        rep(gi, sz(rg[now])) {            int m = rg[now][gi].t;            if (dis[m] == init_value)continue;            if (dis[m] + rg[now][gi].c + cost == dis[t]) {                cost += rg[now][gi].c;                now = m;                break;            }        }        path.push_back(now);    }    rev(path);    return path;}
template<class T> vi dis_path(undigraph<T> &g, vector<T> &dis, int s, int t, int init_value) {    assert(dis[t] != init_value);    int now = t;    vi path;    path.push_back(now);    T cost = 0;    while (now != s) {        rep(gi, sz(g[now])) {            int m = g[now][gi].t;            if (dis[m] == init_value)continue;            if (dis[m] + g[now][gi].c + cost == dis[t]) {                cost += g[now][gi].c;                now = m;                break;            }        }        path.push_back(now);    }    rev(path);    return path;}
template<class T> vi dis_path(digraph<T> &g, vector<vector<T>> &dis, int s, int t, int init_value) { return dis_path(g, dis[s], s, t, init_value); }
template<class T> vi dis_path(undigraph<T> &g, vector<vector<T>> &dis, int s, int t, int init_value) { return dis_path(g, dis[s], s, t, init_value); }
//O(N^2)
template<class T> vector<T> dijkstra_mitu(graph<T> &g, int s, int init_value) {    if (!(0 <= s && s < g.n)) {        debugline("dijkstra_mitu");        deb(s, g.n);        ole();    }    int n = g.n;    int initValue = MAX(int);    vector<T> dis(n, initValue);    dis[s] = 0;    vb used(n);    while (true) {        int v = -1;        rep(u, n) { if (!used[u] && (v == -1 || dis[u] < dis[v]))v = u; }        if (v == -1)break;        if (dis[v] == initValue)break;        used[v] = true;        rep(u, sz(g[v])) {            auto e = g[v][u];            dis[e.t] = min(dis[e.t], dis[v] + e.c);        }    }    /*基本、たどり着かないなら-1*/    for (auto &&d :dis) { if (d == initValue) { d = init_value; }}    return dis;}
template<typename T> struct radixheap {    vector<pair<u64, T> > v[65];    u64 size, last;    radixheap() : size(0), last(0) {}    bool empty() const { return size == 0; }    int getbit(int a) { return a ? 64 - __builtin_clzll(a) : 0; }    void push(u64 key, const T &value) {        ++size;        v[getbit(key ^ last)].emplace_back(key, value);    }    pair<u64, T> pop() {        if (v[0].empty()) {            int idx = 1;            while (v[idx].empty()) ++idx;            last = min_element(begin(v[idx]), end(v[idx]))->first;            for (auto &p : v[idx]) v[getbit(p.first ^ last)].emplace_back(p);            v[idx].clear();        }        --size;        auto ret = v[0].back();        v[0].pop_back();        return ret;    }};
/*radix_heap こっちの方が早い*/
//O((N+M) log N)
vi dijkstra(graph<int> &g, int s, int init_value) {    if (!(0 <= s && s < g.n)) {        debugline("dijkstra");        deb(s, g.n);        ole();    }    /*O((N+M) log N) vs O(N^2)*/    if ((g.n + sz(g.edges)) * log2(N) > g.n * g.n) {        return dijkstra_mitu(g, s, init_value);    }    int initValue = MAX(int);    vi dis(g.n, initValue);    radixheap<int> q;    dis[s] = 0;    q.push(0, s);    while (!q.empty()) {        int nowc, i;        tie(nowc, i) = q.pop();        if (dis[i] != nowc)continue;        for (auto &&e  : g.g[i]) {            int to = e.t;            int c = nowc + e.c;            if (dis[to] > c) {                dis[to] = c;                q.push(dis[to], to);            }        }    }    /*基本、たどり着かないなら-1*/   for (auto &&d :dis) if (d == initValue)d = init_value;    return dis;}
template<class T> vector<T> dijkstra_normal(graph<T> &g, int s, int init_value) {    if (!(0 <= s && s < g.n)) {        debugline("dijkstra");        deb(s, g.n);        ole();    }    if ((g.n + sz(g.edges)) * 20 > g.n * g.n) {        return dijkstra_mitu(g, s, init_value);    }    T initValue = MAX(T);    vector<T> dis(g.n, initValue);    priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> q;    dis[s] = 0;    q.emplace(0, s);    while (q.size()) {        T nowc = q.top().fi;        int i = q.top().se;        q.pop();        if (dis[i] != nowc)continue;        for (auto &&e  : g.g[i]) {            int to = e.t;            T c = nowc + e.c;            if (dis[to] > c) {                dis[to] = c;                q.emplace(dis[to], to);            }        }    }    /*基本、たどり着かないなら-1*/    for (auto &&d :dis) if (d == initValue)d = init_value;    return dis;}
template<class T> vector<T> dijkstra_01(graph<T> &g, int s) {    int N = g.n;    vi dis(N, linf);    dis[s] = 0;    deque<int> q;    q.push_back(s);    vb was(N);    while (!q.empty()) {        int f = q.front();        q.pop_front();        if (was[f])continue;        was[f] = true;        fora(e, g[f]) {            if (dis[e.t] > dis[f] + e.c) {                if (e.c) {                    dis[e.t] = dis[f] + 1;                    q.push_back(e.t);                } else {                    dis[e.t] = dis[f];                    q.push_front(e.t);                }            }        }    }    return dis;}
//dijkstra_cou<mint> : 数える型で書く return vp(dis,cou)
template<class COU,class T=int> auto dijkstra_cou(const graph<T> &g, int s, int init_value) {    if (!(0 <= s && s < g.n)) {        debugline("dijkstra");        deb(s, g.n);        ole();    }    err("count by type COU ");    err("int or mint");    T initValue = MAX(T);    vector<T> dis(g.n, initValue);    vector<COU> cou(g.n);    cou[s] = 1;    priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> q;    dis[s] = 0;    q.emplace(0, s);    while (q.size()) {        T nowc = q.top().fi;        int i = q.top().se;        q.pop();        if (dis[i] != nowc)continue;        for (auto &&e  : g.g[i]) {            int to = e.t;            T c = nowc + e.c;            if (dis[to] > c) {                dis[to] = c;                cou[to] = cou[e.f];                q.emplace(dis[to], to);            }            else if (dis[to] == c) { cou[to] += cou[e.f]; }        }    }    /*基本、たどり着かないなら-1*/    for (auto &&d :dis) if (d == initValue)d = init_value;    return vtop(dis, cou);}
//密グラフの時、warshallに投げる
template<class T> vector<vector<T>> dijkstra_all(const graph<T> &g, int init_value) {int n = g.n;assert(n < 1e4);if (n * n < (n + sz(g.edges)) * 14) {        /*O(N^3) vs O(N (N+M)log N)*/        return warshall(g, init_value); }vector<vector<T>> dis(n);rep(i, n) { dis[i] = dijkstra(g, i, init_value); }return dis;}
//コストを無限に減らせる := -linf
//たどり着けない := linf
template<class T> vector<T> bell(graph<T> &g, int s) {    if (g.n >= 1e4) {        cout << "bell size too big" << endl;        exit(0);    }    vector<T> res(g.n, linf);    res[s] = 0;    vb can(g.n);    /*頂点から行けない頂点を持つ、辺を消しておく */    fix([&](auto ds, int p, int i) -> void {        if (can[i])return;        can[i] = true;        forg(gi, g[i])if (t != p)ds(i, t);    })(-1, 0);    vector<edge<T>> es;    fora(e, g.edges) { if (can[e.f])es += e; }    rep(i, g.n) {        bool upd = false;        fora(e, es) {            if (res[e.f] != linf && res[e.t] > res[e.f] + e.c) {                upd = true;                res[e.t] = res[e.f] + e.c;            }        }        if (!upd)break;    }    rep(i, g.n * 2) {        bool upd = 0;        fora(e, g.edges) {            if (res[e.f] != linf && res[e.t] != -linf && res[e.t] > res[e.f] + e.c) {                upd = 1;                res[e.t] = -linf;            }        }        if (!upd)break;    }    return res;}
//コストを無限に増やせる := linf
//たどり着けない := -linf
template<class T> vector<T> bell_far(graph<T> &g, int s) {    if (g.n >= 1e4) {        cout << "bell_far size too big" << endl;        exit(0);    }    vector<T> res(g.n, linf);    res[s] = 0;    vb can(g.n);    /*頂点から行けない頂点を持つ、辺を消しておく*/    fix([&](auto ds, int p, int i) -> void {        if (can[i])return;        can[i] = true;        forg(gi, g[i])if (t != p)ds(i, t);    })(-1, 0);    vector<edge<T>> es;    fora(e, g.edges) { if (can[e.f])es += e; }    rep(i, g.n) {        bool upd = false;        fora(e, es) {            if (res[e.f] != linf && res[e.t] > res[e.f] - e.c) {/*-c*/                upd = true;                res[e.t] = res[e.f] - e.c;/*-c*/            }        }        if (!upd)break;    }    rep(i, g.n * 2) {        bool upd = 0;        fora(e, g.edges) {            if (res[e.f] != linf && res[e.t] != -linf && res[e.t] > res[e.f] - e.c) {/*-c*/                upd = 1;                res[e.t] = -linf;            }        }        if (!upd)break;    }    rep(i, g.n)res[i] *= -1;    return res;}
template<class T> vector<vector<T>> warshall(const graph<T> &g, int init_value) {    int n = g.n;    assert(n < 1e4);    vector<vector<T> > dis(n, vector<T>(n, linf));    rep(i, n)fora(e, g.g[i]) {            if (dis[e.f][e.t] > e.c) {                dis[e.f][e.t] = e.c;            }        }    rep(i, n)dis[i][i] = 0;    rep(k, n)        rep(i, n)            rep(j, n) {                if (dis[i][j] > dis[i][k] + dis[k][j]) {                    dis[i][j] = dis[i][k] + dis[k][j];                }            }    rep(i, n)rep(j, n) if (dis[i][j] == linf)dis[i][j] = init_value;    return dis;}
template<class T> class MinOp { public: T operator()(T a, T b) { return min(a, b); }};
template<typename OpFunc> struct SparseTable {    OpFunc op;    signed size;    vector<signed> lg;    vector<vector<pair<signed, signed>>> table;    void init(const vector<pair<signed, signed>> &array, OpFunc opfunc) {        signed n = array.size();        op = opfunc;        lg.assign(n + 1, 0);        for (signed i = 1; i <= n; i++) { lg[i] = 31 - __builtin_clz(i); }        table.assign(lg[n] + 1, array);        for (signed i = 1; i <= lg[n]; i++) { for (signed j = 0; j < n; j++) { if (j + (1 << (i - 1)) < n) { table[i][j] = op(table[i - 1][j], table[i - 1][j + (1 << (i - 1))]); } else { table[i][j] = table[i - 1][j]; }}}    }    pair<signed, signed> query(signed l, signed r) {        assert(l < r);        return op(table[lg[r - l]][l], table[lg[r - l]][r - (1 << lg[r - l])]);    }};
struct PMORMQ {    vector<signed> a;    SparseTable<MinOp<pair<signed, signed> > > sparse_table;    vector<vector<vector<signed> > > lookup_table;    vector<signed> block_type;    signed block_size, n_block;    void init(const vector<signed> &array) {        a = array;        signed n = a.size();        block_size = std::max(1, (31 - __builtin_clz(n)) / 2);        while (n % block_size != 0) {            a.push_back(a.back() + 1);            n++;        }        n_block = n / block_size;        vector<pair<signed, signed> > b(n_block, make_pair(INT_MAX, INT_MAX));        for (signed i = 0; i < n; i++) { b[i / block_size] = min(b[i / block_size], make_pair(a[i], i)); }        sparse_table.init(b, MinOp<pair<signed, signed> >());        block_type.assign(n_block, 0);        for (signed i = 0; i < n_block; i++) {            signed cur = 0;            for (signed j = 0; j < block_size - 1; j++) {                signed ind = i * block_size + j;                if (a[ind] < a[ind + 1]) { cur |= 1 << j; }            }            block_type[i] = cur;        }        lookup_table.assign(1 << (block_size - 1), vector<vector<signed> >(block_size, vector<signed>(block_size + 1)));        for (signed i = 0; i < (1 << (block_size - 1)); i++) {            for (signed j = 0; j < block_size; j++) {                signed res = 0;                signed cur = 0;                signed pos = j;                for (signed k = j + 1; k <= block_size; k++) {                    lookup_table[i][j][k] = pos;                    if (i & (1 << (k - 1))) { cur++; } else { cur--; }                    if (res > cur) {                        pos = k;                        res = cur;                    }                }            }        }    }    signed query(signed l, signed r) {        assert(l < r);        signed lb = l / block_size;        signed rb = r / block_size;        if (lb == rb) { return lb * block_size + lookup_table[block_type[lb]][l % block_size][r % block_size]; }        signed pl = lb * block_size + lookup_table[block_type[lb]][l % block_size][block_size];        signed pr = rb * block_size + lookup_table[block_type[rb]][0][r % block_size];        signed pos = pl;        if (r % block_size > 0 && a[pl] > a[pr]) { pos = pr; }        if (lb + 1 == rb) { return pos; }        signed spv = sparse_table.query(lb + 1, rb).second;        if (a[pos] > a[spv]) { return spv; }        return pos;    }};
template<class T=int> class tree : public undigraph<T> {PMORMQ rmq;    int cnt;    vector<signed> id, in;    bool never = true;    bool never_hld = true;    void dfs(int x, int p, int d, int dis = 0) {        id[cnt] = x;        par[x] = p;        rmq_dep.push_back(d);        disv[x] = dis;        in[x] = cnt++;        forg(gi, g[x]) {            if (t == p) { continue; }            dfs(t, x, d + 1, dis + c);            id[cnt] = x;            rmq_dep.push_back(d);            cnt++;        }    }    void precalc() {        never = false;        cnt = 0;        rmq_dep.clear();        disv.assign(n, 0);        in.assign(n, -1);        id.assign(2 * n - 1, -1);        par.assign(n, -1);        dfs(root, -1, 0);        rmq.init(rmq_dep);
#ifdef _DEBUG
        if(n>=100)return;cerr << "---tree---" << endl;        rep(i, n) {            if (!(i == root || sz(g[i]) > 1))continue;            cerr << i << " -> ";            vi ts;            forg(gi, g[i]) { if (t != par[i])ts.push_back(t); }            rep(i, sz(ts) - 1)cerr << ts[i] << ", ";            if(sz(ts))cerr << ts.back() << endl;        }        cerr << endl;
#endif
    }    int pos;    void hld_build() {        never_hld = false;        if (never)precalc();        g.resize(n);        vid.resize(n, -1);        head.resize(n);        heavy.resize(n, -1);        depth.resize(n);        inv.resize(n);        subl.resize(n);        subr.resize(n);        dfs(root, -1);        t = 0;        dfs_hld(root);
#ifdef _DEBUG
        if (n >= 100)return;        cerr << "---hld_index---" << endl;        vi inds;        rep(i, n) if (sz(g[i]))inds.push_back(i);        rep(i, sz(inds)) {            str s = tos(bel(inds[i]));            cerr << std::right << std::setw(sz(s) + (i ? 1 : 0)) << inds[i];        }        cerr << endl;        rep(i, sz(inds)) { cerr << bel(inds[i]) << " "; }        cerr << endl << endl;        cerr << "---hld_edge_index---" << endl;        fora(e, edges) { if (e.f <= e.t) cerr << e.f << "-" << e.t << " " << bel(e) << endl; }        cerr << endl << endl;        cerr << "!!query!! edge or not edge carefull!!" << endl;
#endif
    }    int dfs(int curr, int prev) {        int sub = 1, max_sub = 0;        heavy[curr] = -1;        forg(i, g[curr]) {            int next = t;            if (next != prev) {                depth[next] = depth[curr] + 1;                int sub_next = dfs(next, curr);                sub += sub_next;                if (max_sub < sub_next) max_sub = sub_next, heavy[curr] = next;            }        }        return sub;    }    int t = 0;    void dfs_hld(int v = 0) {        vid[v] = subl[v] = t;        t++;        inv[subl[v]] = v;        if (0 <= heavy[v]) {            head[heavy[v]] = head[v];            dfs_hld(heavy[v]);        }        forg(i, g[v])if (depth[v] < depth[t])                if (t != heavy[v]) {                    head[t] = t;                    dfs_hld(t);                }        subr[v] = t;    }vector<signed> rmq_dep;
public:
    using undigraph<T>::g;    using undigraph<T>::n;    using undigraph<T>::edges;
    vector<int> disv;
    //部分木の  [左端、右端)  index
    //部分木の辺に加算する場合
    //add(subl[i],subr[i],x)
    //add(sub[i],sub[i+1],-x)
    vector<int> vid, head, heavy, par, depth, inv, subl, subr;
    int root;
    tree(int n_, int root = 0) : undigraph<T>(n_), root(root) { n = n_; }
    int lca(int a, int b) {        if (never)precalc();        int x = in[a];        int y = in[b];        if (x > y) { swap(x, y); }        int pos = rmq.query(x, y + 1);        return id[pos];    }
    int dis(int a, int b) {        if (never)precalc();        int x = in[a];        int y = in[b];        if (x > y) { swap(x, y); }        int pos = rmq.query(x, y + 1);        int p = id[pos];        return disv[a] + disv[b] - disv[p] * 2;    }
    /*O(N) hldを使わず木を普通にたどる*/
    void for_each_l(int u, int v, function<void(int)> fnode) {        int r = lca(u, v);        while (u != r) {            fnode(u);            u = par[u];        }        while (v != r) {            fnode(v);            v = par[v];        }        fnode(r);    }
    void for_each_edge_l/*O(N) 頂点に対しての処理順が可換*/(int u, int v, function<void(edge<int> &)> fedge) {        int r = lca(u, v);        auto sub = [&](int u, int r) {            while (u != r) {                forg(gi, g[u]) {                    if (t == par[u]) {                        fedge(g[u][gi]);                        u = par[u];                        break;                    }                }            }        };        sub(u, r);        sub(v, r);    }
    //Fは半開 (u,v)は木の頂点
    //中ではhldの頂点を見るため、seg木のupdateはhldのindexで行なう
    void for_each_/*[l,r)*/(int u, int v, const function<void(int, int)> &f) {        if (never_hld)hld_build();        while (1) {            if (vid[u] > vid[v]) swap(u, v);            int l = max(vid[head[v]], vid[u]);            int r = vid[v] + 1;            f(l, r);            if (head[u] != head[v]) v = par[head[v]]; else break;        }    }
    void for_each_edge/*[l,r) O(log(N)) 辺を頂点として扱っている 上と同じ感じで使える*/(int u, int v, const function<void(int, int)> &f) {        if (never_hld)hld_build();        while (1) {            if (vid[u] > vid[v]) swap(u, v);            if (head[u] != head[v]) {                int l = vid[head[v]];                int r = vid[v] + 1;                f(l, r);                v = par[head[v]];            } else {                if (u != v) {                    int l = vid[u] + 1;                    int r = vid[v] + 1;                    f(l, r);                }                break;            }        }    }
    int bel(int v) {        /*hld内での頂点番号を返す*/        if (never_hld)hld_build();return vid[v];}
    //下の頂点に辺のクエリを持たせている
    int bel(int f, int t) {        /*辺のクエリを扱うときどの頂点に持たせればいいか(vidを返すのでそのままupd出来る)*/        if (never_hld)hld_build();return depth[f] > depth[t] ? vid[f] : vid[t];}
    int bel(edge<T> &e) {        /*辺のクエリを扱うときどの頂点に持たせればいいか(vidを返すのでそのままupd出来る)*/        if (never_hld)hld_build();return depth[e.f] > depth[e.t] ? vid[e.f] : vid[e.t];}
    template<class ... U> int operator()(U ... args) { return bel(args...); }
    //path l -> r += v
    template<class S> void seg_add(S &seg, int lhei, int rhei, int v) {for_each_(lhei, rhei, [&](int l, int r) { seg.add(l, r, v); });}
    template<class S> void seg_update(S &seg, int lhei, int rhei, int v) {for_each_(lhei, rhei, [&](int l, int r) { seg.update(l, r, v); });}
    template<class S> T seg_get(S &seg, int lhei, int rhei) {        T res = seg.e;        for_each_(lhei, rhei, [&](int l, int r) { res = seg.f(res, seg.get(l, r)); });        return res;    }
    template<class S> void seg_add_edge(S &seg, int lhei, int rhei, int v) { for_each_edge(lhei, rhei, [&](int l, int r) { seg.add(l, r, v); }); }
    template<class S> void seg_update_edge(S &seg, int lhei, int rhei, int v) { for_each_edge(lhei, rhei, [&](int l, int r) { seg.update(l, r, v); }); }
    template<class S> T seg_get_edge(S &seg, int lhei, int rhei) {        T res = seg.e;        for_each_edge(lhei, rhei, [&](int l, int r) { res = seg.f(res, seg.get(l, r)); });        return res;    }
    //単体   edgeは上で処理できる
    template<class S> void seg_add(S &seg, int i, int v) { seg.add(bel(i), v); }
    template<class S> void seg_update(S &seg, int i, int v) { seg.update(bel(i), v); }
    template<class S> T seg_get(S &seg, int i) { return seg[i]; }
    template<class S> void seg_del(S &seg, int i) { seg.del(bel(i)); }

    //部分木iに対するクエリ
    template<class S> void seg_add_sub(S &seg, int i, int v) {if (never_hld)hld_build();seg.add(subl[i], subr[i], v);}
    template<class S> void seg_update_sub(S &seg, int i, int v) {if (never_hld)hld_build();seg.update(subl[i], subr[i], v);}
    template<class S> T seg_get_sub(S &seg, int i, int v) {if (never_hld)hld_build();return seg.get(subl[i], subr[i], v);}
    template<class S> void seg_add_sub_edge(S &seg, int i, int v) {if (never_hld)hld_build();/*iの上の辺が数えられてしまうため、i+1から*/seg.add(subl[i] + 1, subr[i], v);}
    template<class S> void seg_update_sub_edge(S &seg, int i, int v) {if (never_hld)hld_build();/*iの上の辺が数えられてしまうため、i+1から*/seg.update(subl[i] + 1, subr[i], v);}
    template<class S> T seg_get_sub_edge(S &seg, int i, int v) {if (never_hld)hld_build();/*iの上の辺が数えられてしまうため、i+1から*/return seg.get(subl[i] + 1, subr[i], v);}
};
//辺が多いのでedgesを持たない
//cost oo, ox, xo, xx 渡す
template<class T=int> class grid_k6 : public undigraph<T> {
    vi costs;
public:
    using undigraph<T>::g;    using undigraph<T>::n;    using undigraph<T>::edges;
    int H, W;
    vector<vector<char>> ba;
    char wall;

    void add(int f, int t, T c = 1){
        cout << "grid" << endl;
        if (!(0 <= f && f < n && 0 <= t && t < n)) {            debugline("grid_k6 add");            deb(f, t, c);            ole();        }
        g[f].emplace_back(f, t, c);
        g[t].emplace_back(t, f, c);
        //undigraphと違い、edgesを持たない
    }
    int getid(int h, int w) {        assert(ins(h, w, H, W));        return W * h + w;    }
    int getid(P p) { return getid(p.first, p.second); }
    P get2(int id) { return mp(id / W, id % W); }
    P operator()(int id) { return get2(id); }
    int operator()(int h, int w) { return getid(h, w); }
    int operator()(P p) { return getid(p); }
    //辺は無い
    grid_k6(int H, int W) : H(H), W(W), undigraph<T>(H * W) {}
    grid_k6(vector<vector<char>> ba, char wall = '#') : H(sz(ba)), W(sz(ba[0])), undigraph<T>(sz(ba) * sz(ba[0])) {        rep(h, H) {            rep(w, W) {                if (ba[h][w] == wall)con;                int f = getid(h, w);                if (w + 1 < W && ba[h][w + 1] != wall) { add(f, getid(h, w + 1)); }                if (h + 1 < H && ba[h + 1][w] != wall) { add(f, getid(h + 1, w)); }            }        }    }
    /*o -> o, o -> x, x-> x*/
    grid_k6(vector<vector<char>> ba, int oo, int ox, int xo, int xx, char wall = '#') : H(sz(ba)), W(sz(ba[0])), undigraph<T>(sz(ba) * sz(ba[0])), costs({oo, ox, xo, xx}), ba(ba), wall(wall) {        rep(h, H) {            rep(w, W) {                add(h, w, h + 1, w);                add(h, w, h - 1, w);                add(h, w, h, w + 1);                add(h, w, h, w - 1);            }        }    }
    void add(int fh, int fw, int th, int tw) {        if (ins(fh, fw, H, W) && ins(th, tw, H, W)) {            int cm = 0;            if (ba[fh][fw] == wall) { cm += 2; }            if (ba[th][tw] == wall) { cm++; }            int f = getid(fh, fw);            int t = getid(th, tw);            g[f].emplace_back(f, t, costs[cm]);        }    }
    void set_edges() { rep(i, n)fora(e, g[i])if (e.f < e.t)edges.push_back(e); }
};
//辺によりメモリを大量消費ためedgesを消している
//頂点10^6でメモリを190MB(制限の8割)使う
//左上から右下に移動できる
template<class T=int> class digrid_k6 : public digraph<T> {
public:
    using digraph<T>::g;    using digraph<T>::n;    using digraph<T>::edges;
    int H, W;
    void add(int f, int t, T c = 1) {
        if (!(0 <= f && f < n && 0 <= t && t < n)) {            debugline("digrid_k6 add");            deb(f, t, c);            ole();        }
        g[f].emplace_back(f, t, c);
        /*digraphと違いedgesを持たない*/
    }
    int getid(int h, int w) {        if (!ins(h, w, H, W))return -1;        return W * h + w;    }
    P get2(int id) { return mp(id / W, id % W); }
    P operator()(int id) { return get2(id); }
    int operator()(int h, int w) { return getid(h, w); }
    digrid_k6(int H, int W) : H(H), W(W), digraph<T>(H * W){}
    digrid_k6(vector<vector<char>> ba, char wall = '#') : H(sz(ba)), W(sz(ba[0])), digraph<T>(sz(ba) * sz(ba[0])) {        rep(h, H) {            rep(w, W) {                if (ba[h][w] == wall)con;                int f = getid(h, w);                if (w + 1 < W && ba[h][w + 1] != wall) { add(f, getid(h, w + 1)); }                if (h + 1 < H && ba[h + 1][w] != wall) { add(f, getid(h + 1, w)); }            }        }    }
    void add(int fh, int fw, int th, int tw) { add(getid(fh, fw), getid(th, tw)); }
    void set_edges() { rep(i, n)fora(e, g[i])edges.push_back(e); }
};
//edgesを持たない
//dijkstra(g,0)[t] とかける (t+n-1等とする必要はない)
template<class T=int>class segdi : public digraph<T>{
    int getid(int k){
        if(k >= len*2-1){
            return k;
        }
        else if(k < len - 1){
            return k + len;
        }else{
            return k - len + 1;
        }
    }
    void add(int f, int t, T c = 1) {
        f = getid(f);
        t = getid(t);
        if (!(0 <= f && f < n && 0 <= t && t < n)) {            debugline("segdi add");            deb(f, t, c);            ole();        }
        g[f].emplace_back(f, t, c);
        /*digraphと違いedgesを持たない*/
    }/*@formatter:on*/
    int mid;
    int len;
public:
    using digraph<T>::g;    using digraph<T>::n;    using digraph<T>::edges;
    //頂点が足りなくなったらresize
    segdi(int n_) : digraph<T>(1) {
        int nn = 1;
        while (nn < n_)nn *= 2;
        n_ = nn;
        len = n_;
        this->resize(n_ + (n_ - 1) * 2 + n_);
        mid = n_ + (n_ - 1) * 2;
        int ad = len * 2 - 1;
        rep(i, len - 1) {
            add(i, i * 2 + 1, 0);
            add(i, i * 2 + 2, 0);
            if (i * 2 + 1 >= len - 1) {
                add(i * 2 + 1, i + ad, 0);
                add(i * 2 + 2, i + ad, 0);
            } else {
                add(i * 2 + 1 + ad, i + ad, 0);
                add(i * 2 + 2 + ad, i + ad, 0);
            }
        }
    }
    void dfs(vi &list, int nl, int nr, int k, int l, int r) {
        if (r <= nl || nr <= l)return;
        if (l <= nl && nr <= r) {
            list += k;
        } else {
            dfs(list, nl, (nl + nr) / 2, k * 2 + 1, l, r);
            dfs(list, (nl + nr) / 2, nr, k * 2 + 2, l, r);
        }
    }
    void rekkyo(vi &list, int l, int r) {
        l += len - 1;
        r += len - 1;
        while (l < r) {
            if (!(l & 1)) { list.push_back(l); }
            if (!(r & 1)) { list.push_back(r - 1); }
            l >>= 1;
            r = (r - 1) >> 1;
        }
    }
    //半開
    void add(int fl, int fr, int tl, int tr, int cost) {
        /*fは下側*/
        int ad = 2 * len - 1;
        if (mid >= n) { this->resize(n + 100); }
        {
            vi fs;
            int l = fl, r = fr;
            l += len - 1;
            r += len - 1;
            while (l < r) {
                if (!(l & 1)) {
                    fs.push_back(l);
                }
                if (!(r & 1)) { fs.push_back(r - 1); }
                l >>= 1;
                r = (r - 1) >> 1;
            }
            fora(f, fs) {
                if (f - len + 1 < 0)add(f + ad, mid, cost);
                else add(f, mid, cost);
            }
        }
        {
            int l = tl, r = tr;
            l += len - 1;
            r += len - 1;
            while (l < r) {
                if (!(l & 1)) { add(mid, l, 0); }
                if (!(r & 1)) { add(mid, r - 1, 0); }
                l >>= 1;
                r = (r - 1) >> 1;
            }
        }
        mid++;
    }

};/*@formatter:off*/
//@出力
template<class T> ostream &operator<<(ostream &os, digraph<T> &g) {    os << endl << g.n << " " << sz(g.edges) << endl;    fore(gi, g.edges) { os << f << " " << t << " " << c << endl; }    return os;}template<class T> ostream &operator<<(ostream &os, undigraph<T> &g) {    os << endl << g.n << " " << sz(g.edges) << endl;    fore(gi, g.edges) { if (f < t)os << f << " " << t << " " << c << endl; }    return os;}

//@判定
template<class T> bool nibu(const graph<T> &g) {int size = 0;    rep(i, g.n)size += sz(g.g[i]);    if (size == 0)return true;    unionfind uf(g.n * 2);    rep(i, g.n)fora(e, g.g[i])uf.unite(e.f, e.t + g.n), uf.unite(e.f + g.n, e.t);    rep(i, g.n)if (uf.same(i, i + g.n))return 0;    return 1;}
    //二部グラフを色分けした際の頂点数を返す
template<class T> vp nibug(graph<T> &g) {    vp cg;    if (!nibu(g)) {        debugline("nibu");        ole();    }    int n = g.size();    vb was(n);    queue<P> q;    rep(i, n) {        if (was[i])continue;        q.push(mp(i, 1));        was[i] = 1;        int red = 0;        int coun = 0;        while (q.size()) {            int now = q.front().fi;            int col = q.front().se;            red += col;            coun++;            q.pop();            forg(gi, g[now]) {                if (was[t])continue;                q.push(mp(t, col ^ 1));                was[t] = 1;            }        }        cg.push_back(mp(red, coun - red));    }    return cg;}
    //連結グラフが与えられる 閉路があるか
template<class T> bool close(undigraph<T> &g) {    int n = 0;    int e = 0;    rep(i, g.n) {        if (sz(g[i]))n++;        forg(gi, g[i]) { e++; }    }    return (e >> 1) >= n;}
template<class T> bool close(undigraph<T> &g, int v) {    unionfind uf(g.n);    rep(i, g.n) {        forg(gi, g[i]) {            if (f < t)break;            if (f == t && f == v)return true;            if (uf.same(f, v) && uf.same(t, v))return true;            uf.unite(f, t);        }    }    return false;}template<class T> bool close(digraph<T> &g) {    vi res;    return topo(res, g);}

//@変形
//条件(f!=0等 f,t,cが使える)を満たすsubgraphをg2に代入
#define sub_di(g, g2, zhouken) g2.resize(g.n);fore(gi, g.edges){if(zhouken){g2.add(f,t,c);}}
#define sub_un(g, g2, zhouken) g2.resize(g.n);fore(gi, g.edges){if(zhouken && f < t){g2.add(f,t,c);}}
    //閉路がなければtrue
bool topo(vi &res, digraph<int> &g) {    int n = g.g.size();    vi nyu(n);    rep(i, n)for (auto &&e :g[i])nyu[e.t]++;    queue<int> st;    rep(i, n)if (nyu[i] == 0)st.push(i);    while (st.size()) {        int v = st.front();        st.pop();        res.push_back(v);        fora(e, g[v]) if (--nyu[e.t] == 0)st.push(e.t);    }    return res.size() == n;}
    //辞書順最小トポロジカルソート
bool topos(vi &res, digraph<int> &g) {    int n = g.g.size();    vi nyu(n);    rep(i, n)for (auto &&e :g[i])nyu[e.t]++;    /*小さい順*/    priority_queue<int, vector<int>, greater<int> > q;    rep(i, n)if (nyu[i] == 0)q.push(i);    while (q.size()) {        int i = q.top();        q.pop();        res.push_back(i);        fora(e, g[i])if (--nyu[e.t] == 0)q.push(e.t);    }    return res.size() == n;}
template<class T> digraph<T> rev(digraph<T> &g) {    digraph<T> r(g.n);    rep(i, g.n) { forg(gi, g[i]) { r.add(t, f, c); }}    return r;}
    //lc,rcは子を持つ中で一番左、右
    //(g,ind,l,r)
template<class T> tree<T> get_bfs_tree(tree<T> &g, vi &ind, vi &l, vi &r) {if (sz(ind)) {cerr << "ind must be empty" << endl;exit(0);}int N = sz(g);ind.resize(N);l.resize(N, inf);r.resize(N, -1);tree<T> h(N);queue<P> q;q.emplace(-1, 0);int c = 0;while (sz(q)) {int p = q.front().first;int i = q.front().second;q.pop();ind[i] = c;if (~p)chmi(l[ind[p]], c);if (~p)chma(r[ind[p]], c);c++;forg(gi, g[i]) {if (t != p)q.emplace(i, t);}}fora(e, g.edges) {if (e.f < e.t) {h.add(ind[e.f], ind[e.t], e.c);}}rep(i, N) {if (l[i] == inf)l[i] = -1;}return h;}
    //lc,rcは子を持つ中で一番左、右
    // たとえばl[lc[x]は2段下の最左
    //(g,ind,l,r,lc,rc)
template<class T> tree<T> get_bfs_tree(tree<T> &g, vi &ind, vi &l, vi &r, vi &lc, vi &rc) {    if (sz(ind)) {        cerr << "ind must be empty" << endl;        exit(0);    }    int N = sz(g);    ind.resize(N);    l.resize(N, inf);    lc.resize(N, inf);    r.resize(N, -1);    rc.resize(N, -1);    tree<T> h(N);    queue<P> q;    q.emplace(-1, 0);    int c = 0;    while (sz(q)) {        int p = q.front().first;        int i = q.front().second;        q.pop();        ind[i] = c;        if (~p) {            chmi(l[ind[p]], c);            chma(r[ind[p]], c);            if (sz(g[i]) > 1) {                chmi(lc[ind[p]], c);                chma(rc[ind[p]], c);            }        }        c++;        forg(gi, g[i]) {            if (t != p)q.emplace(i, t);        }    }    fora(e, g.edges) {        if (e.f < e.t) {            h.add(ind[e.f], ind[e.t], e.c);        }    }    rep(i, N) {        if (l[i] == inf)l[i] = -1;        if (lc[i] == inf)lc[i] = -1;    }    return h;}

//@集計
template<class T> vi indegree(graph<T> &g) {vi ret(g.size());rep(i, g.size()) { forg(gi, g[i]) { ret[t]++; }}return ret;}
template<class T> vi outdegree(graph<T> &g) {vi ret(g.size());rep(i, g.size()) { ret[i] = g[i].size(); }return ret;}
#define kansetu articulation
P farthest(undigraph<> &E, int cur, int pre, int d, vi &D) {    D[cur] = d;    P r = {d, cur};    forg(gi, E[cur]) if (t != pre) {            P v = farthest(E, t, cur, d + 1, D);            r = max(r, v);        }    return r;}
//dagでなければ-1を返す
int diameter(digraph<> &g) {    vi per;    if (!topo(per, g))return -1;    int n = g.n;    vi dp(n, 1);    fora(v, per) { forg(gi, g[v]) { chma(dp[t], dp[f] + 1); }}    return max(dp);}
//iから最も離れた距離
vi diameters(undigraph<> &E) { /* diameter,center*/vi D[3];    D[0].resize(E.size());    D[1].resize(E.size());    auto v1 = farthest(E, 0, 0, 0, D[0]);    auto v2 = farthest(E, v1.second, v1.second, 0, D[0]);    farthest(E, v2.second, v2.second, 0, D[1]);    int i;    rep(i, D[0].size()) D[2].push_back(max(D[0][i], D[1][i]));    return D[2];}
int diameter(undigraph<> &E) {vi d = diameters(E);return max(d);}
//i d
vp diameter_p(undigraph<> &E) { /* diameter,center*/vector<int> D[3];    D[0].resize(E.size());    D[1].resize(E.size());    auto v1 = farthest(E, 0, 0, 0, D[0]);    auto v2 = farthest(E, v1.second, v1.second, 0, D[0]);    farthest(E, v2.second, v2.second, 0, D[1]);    int i;    vp res(E.size());    rep(i, D[0].size()) { if (D[0][i] > D[1][i])res[i] = mp(D[0][i], v1.second); else res[i] = mp(D[1][i], v2.second); }    return res;}

//@列挙 取得
//閉路がある時linfを返す
template<class T> int longest_path(digraph<T> &g) {    vi top;    if (!topo(top, g)) { return linf; }    int n = sz(top);    vi dp(n, 0);    for (auto &&i : top) { forg(gi, g[i]) { chma(dp[t], dp[i] + 1); }}    return max(dp);}
template<class T> vi longest_path_v(digraph<T> &g) {    vi top;    if (!topo(top, g)) { return vi(); }    int n = sz(top);    vi dp(n, 0);    vi pre(n, -1);    for (auto &&i : top) { forg(gi, g[i]) { if (chma(dp[t], dp[i] + 1)) { pre[t] = i; }}}    int s = std::max_element(dp.begin(), dp.end()) - dp.begin();    vi path;    while (s != -1) {        path.push_back(s);        s = pre[s];    }    std::reverse(path.begin(), path.end());    return path;}
//橋を列挙する (取り除くと連結でなくなる辺)
template<class T> vp bridge(graph<T> &G) {    static bool was;    vp brid;    vi articulation;    vi ord(G.n), low(G.n);    vb vis(G.n);    function<void(int, int, int)> dfs = [&](int v, int p, int k) {        vis[v] = true;        ord[v] = k++;        low[v] = ord[v];        bool isArticulation = false;        int ct = 0;        for (int i = 0; i < G[v].size(); i++) {            if (!vis[G[v][i].t]) {                ct++;                dfs(G[v][i].t, v, k);                low[v] = min(low[v], low[G[v][i].t]);                if (~p && ord[v] <= low[G[v][i].t]) isArticulation = true;                if (ord[v] < low[G[v][i].t]) brid.push_back(make_pair(min(v, G[v][i].t), max(v, G[v][i].t)));            } else if (G[v][i].t != p) { low[v] = min(low[v], ord[G[v][i].t]); }        }        if (p == -1 && ct > 1) isArticulation = true;        if (isArticulation) articulation.push_back(v);    };    int k = 0;    rep(i, G.n) { if (!vis[i]) dfs(i, -1, k); }    sort(brid.begin(), brid.end());    return brid;}
//間接点を列挙する (取り除くと連結でなくなる点)
template<class T> vi articulation(undigraph<T> &G) {    static bool was;    vp bridge;    vi arti;    vi ord(G.n), low(G.n);    vb vis(G.n);    function<void(int, int, int)> dfs = [&](int v, int p, int k) {        vis[v] = true;        ord[v] = k++;        low[v] = ord[v];        bool isArticulation = false;        int ct = 0;        for (int i = 0; i < G[v].size(); i++) {            if (!vis[G[v][i].t]) {                ct++;                dfs(G[v][i].t, v, k);                low[v] = min(low[v], low[G[v][i].t]);                if (~p && ord[v] <= low[G[v][i].t]) isArticulation = true;                if (ord[v] < low[G[v][i].t]) bridge.push_back(make_pair(min(v, G[v][i].t), max(v, G[v][i].t)));            } else if (G[v][i].t != p) { low[v] = min(low[v], ord[G[v][i].t]); }        }        if (p == -1 && ct > 1) isArticulation = true;        if (isArticulation) arti.push_back(v);    };    int k = 0;    rep(i, G.n) { if (!vis[i]) dfs(i, -1, k); }    sort(arti.begin(), arti.end());    return arti;}
//閉路パスを一つ返す
vi close_path(digraph<> &g) {    int n = g.n;    vi state(n);    vi path;    rep(i, n) if (!state[i]) {            if (fix([&](auto dfs, int v) -> bool {                if (state[v]) {                    if (state[v] == 1) {                        path.erase(path.begin(), find(path.begin(), path.end(), v));                        return true;                    }                    return false;                }                path.push_back(v);                state[v] = 1;                forg(gi, g[v]) { if (dfs(t))return true; }                state[v] = -1;                path.pop_back();                return false;            })(i)) { return path; }        }    return vi();}
vi close_path_min(digraph<> &g) {    int n = g.n;    vvi(dis, n);    rep(i, n)dis[i] = dijkstra(g, i, linf);    int mind = linf;    int f = 0, t = 0;    rep(i, n) {        rep(j, n) {            if (i == j)continue;            if (chmi(mind, dis[i][j] + dis[j][i])) {                f = i;                t = j;            }        }    }    vi path;    auto add = [&](int f, int t) {        int now = f;        while (now != t) {            rep(i, n) {                if (dis[now][i] == 1 && dis[f][i] + dis[i][t] == dis[f][t]) {                    path.push_back(i);                    now = i;                    break;                }            }        }    };    add(f, t);    add(t, f);    return path;}
//iを含む最短閉路 https://atcoder.jp/contests/abc022/tasks/abc022_c
/*閉路が1つしかない場合、その閉路に含まれる頂点を1としたvectorを返す*/;
template<class T> vi get_close1(digraph<T> &g) {    int n = g.n;    queue<int> q;    vi d = outdegree(g);    vi res(n, 1);    rep(i, n) {        if (d[i] == 0) {            q += i;            res[i] = 0;        }    }    auto rg = rev(g);    while (q.size()) {        auto now = q.front();        q.pop();        forg(gi, rg[now]) {            if (--d[t] == 0) {                q += t;                res[t] = 0;            }        }    }    return res;}

//@アルゴリズム
template<class T> int krus(undigraph<T> &g) {    int res = 0;    unionfind uf(g.n);    if (sz(g.edges) == 0)g.set_edges();    int i = 0;    auto E = g.edges;    sort(E);    fora(e, E) { if (uf.unite(e.f, e.t)) { res += e.c; }}    return res;}
template<class T> vector<edge<T>> krus_ed(undigraph<T> &g) {    unionfind uf(g.n);    if (sz(g.edges) == 0)g.set_edges();    int i = 0;    auto E = g.edges;    sort(E);    vector<edge<T>> res;    fora(e, E) { if (uf.unite(e.f, e.t)) { res.push_back(e); }}    return res;}
template<class T> tree<T> krus_tr(undigraph<T> &g) {    tree<T> res(g.n);    unionfind uf(g.n);    if (sz(g.edges) == 0)g.set_edges();    int i = 0;    auto E = g.edges;    sort(E);    fora(e, E) { if (uf.unite(e.f, e.t)) { res.add(e.f, e.t); }}    return res;}

//@実験
digraph<> rang_di(int n, int m, bool zibun = 0, bool taju = 0) {    umapp was;    digraph<> g(n);    was[mp(-1, -2)] = 1;    while (m) {        int f = -1, t = -2;        while (f < 0 || (!taju && was[mp(f, t)])) {            f = rand(0, n - 1);            t = rand(0, n - 1);            if (!zibun && f == t)f = -1;        }        g.add(f, t);        was[mp(f, t)] = 1;        m--;    }    return g;}
digraph<> perfect_di(int n, bool zibun = 0) {    digraph<> g(n);    rep(i, n) {        rep(j, n) {            if (!zibun && i == j)con;            g.add(i, j);        }    }    return g;}
undigraph<> rang_un(int n, int m, bool zibun = 0, bool taju = 0) {    umapp was;    undigraph<> g(n);    was[mp(-1, -2)] = 1;    while (m) {        int f = -1, t = -2;        while (f < 0 || (!taju && was[mp(min(f, t), max(f, t))])) {            f = rand(0, n - 1);            t = rand(0, n - 1);            if (!zibun && f == t)f = -1;        }        g.add(f, t);        was[mp(min(f, t), max(f, t))] = 1;        m--;    }    return g;}
undigraph<> perfect_un(int n, bool zibun = 0){    undigraph<> g(n);    rep(i, n) {        rep(j, i, n) {            if (!zibun && i == j)con;            g.add(i, j);        }    }    return g;}
/*頂点数がkの木を一つ返す サイズが0の木が帰ったら終了*/
tree<int> next_tree(int k) {    assert(2 <= k && k < 11);    static str name;    static ifstream ina;    static int rem;    static vp edges;    static int pk = -1;/*前回見たk*/    if (pk != k) {        if (~pk)ina.close();        edges.clear();        pk = k;        name = (k == 6) ? "C:\\Users\\kaout\\Desktop\\trees_sizek\\nazeka6.txt" : "C:\\Users\\kaout\\Desktop\\trees_sizek\\tree_size" + tos(k) + ".txt";        ina = ifstream(name);        rem = pow(k, k - 2);/*Cayleyの定理*/        rep(i, k)rep(j, i + 1, k)edges.emplace_back(i, j);        pk = k;    }    tree<int> g(k);    if (rem == 0) {        g.resize(0);        return g;    }    int m;    ina >> m;    while (m) {        int lb = lbit(m);        int id = log2(lb);        g.add(edges[id].first, edges[id].second);        m ^= lb;    }    rem--;    return g;}
undigraph<int> next_undi(int k) {    assert(2 <= k && k < 9);    static str name;    static ifstream ina;    static int rem;    static vp edges;    static vi lims = {-1, -1, 1, 4, 38, 728, 26704, 1866256};    static int pk = -1;/*前回見たk*/    if (pk != k) {        if (~pk)ina.close();        edges.clear();        pk = k;        name = (k == 6) ? "C:\\Users\\kaout\\Desktop\\undi_sizek\\roku.txt" : "C:\\Users\\kaout\\Desktop\\undi_sizek\\undi_size" + tos(k) + ".txt";        ina = ifstream(name);        rem = lims[k];        rep(i, k)rep(j, i + 1, k)edges.emplace_back(i, j);        pk = k;    }    undigraph<int> g(k);    if (rem == 0) {        g.resize(0);        return g;    }    int m;    ina >> m;    while (m) {        int lb = lbit(m);        int id = log2(lb);        g.add(edges[id].first, edges[id].second);        m ^= lb;    }    rem--;    return g;}
vector<tree<int>> trees(int k) {    vector<tree<int>> res;    while (1) {        tree<int> g = next_tree(k);        if (sz(g) == 0)break;        res.push_back(g);    }    return res;}
vector<undigraph<int>> undis(int k) {    vector<undigraph<int>> res;    while (1) {        undigraph<int> g = next_undi(k);        if (sz(g) == 0)break;        res.push_back(g);    }    return res;}
template<class T> vector<vector<int>> table(graph<T> &g,int init_value) {    vvi(res, g.n, g.n,init_value);    rep(i, g.n) { forg(gi, g[i]) { res[i][t] = c; }}    return res;}
//type,idが使いたい場合はgraty
/*@formatter:on*/

void solve() {
    in(N, M);
    segdi<> g(N);
    rep(i, M) {
        din(l, r, c);
        l--;
        g.add(l, r, l, r, c);
    }
    out(dijkstra(g, 0, -1)[N - 1]);
}
//(len, init_value)
//(vector<> &) 参照を付けないとおかしくなる
//答えが存在しないことがある場合 index付きで、答えが-1になるかで判定したほうがいい
template<class M=int> struct segmin {/*@formatter:off*/
    int n;
    vector<M> seg;
    M f(M a, M b) { return a < b ? a : b; }
    const M e = MAX(M);
    segmin(int len, M init_value = MAX(M)) : e(init_value) {        n = 1;        int asz = len;        while (n <= asz)n <<= 1;        seg.assign((n << 1) - 1, e);        rep(i, asz) seg[i + n - 1] = e;        rer(i, n - 2)seg[i] = f(seg[(i << 1) | 1], seg[(i << 1) + 2]);        debu();    }
    segmin(vector<M> &dat, M init_value = MAX(M)) : e(init_value) {        n = 1;        int asz = dat.size();        while (n <= asz)n <<= 1;        seg.assign((n << 1) - 1, e);        rep(i, asz) seg[i + n - 1] = dat[i];        rer(i, n - 2)seg[i] = f(seg[(i << 1) | 1], seg[(i << 1) + 2]);        debu();    }
private:
    //[]を使う
    void update(int k, M v, bool debug = true) {        int tk = k;        seg[k + n - 1] = v;        k += n - 1;        while (k) {            k = (k - 1) >> 1;            seg[k] = f(seg[(k << 1) | 1], seg[(k << 1) + 2]);        }        if (debug)debup(tk, v);    }
    M query(int a, int b, int k, int l, int r) {        if (r <= a || b <= l)return e;        else if (a <= l && r <= b)return seg[k];        else {            M sl = query(a, b, (k << 1) | 1, l, (l + r) >> 1);            M sr = query(a, b, (k << 1) + 2, (l + r) >> 1, r);            return f(sl, sr);        }    }
public:
    void del(int k) {advance();update(k, e);}
    void add(int k, int v, bool debug = true) {advance();update(k, v + seg[k + n - 1], debug);}
/*@formatter:on*/
    M query_bottomup(int l, int r, bool debug = true) {
        advance();
        int tl = l, tr = r;
        M retl = e;
        M retr = e;
        l += n - 1;
        r += n - 1;
        while (l < r) {
            /*右なら、次に右上へ移動してしまうので*/
            if (!(l & 1)) { retl = f(retl, seg[l]); }
            /*右なら、左を取らないと機会を逃す*/            if (!(r & 1)) { retr = f(seg[r - 1], retr); }
            l >>= 1;
            r = (r - 1) >> 1;
        }
        M ret = f(retl, retr);
        if (debug)debge(tl, tr, ret);
        return ret;
    }
    /*    M get(int a = 0, int b = -1) { return query(a, b == -1 ? n : b, 0, 0, n); }*/
    M get(int a, int b, bool debug = true) { return query_bottomup(a, min(b, n), debug); }
    M get(int b = inf, bool debug = true) { return query_bottomup(0, min(b, n), debug); }
    M operator()(int b = inf) { return get(0, b); }
    M operator()(int a, int b) { return get(a, b); }
    queue<pair<int, M>> q;
    M &operator[](int k) {
        q.push(mp(k, seg[k + n - 1]));
        return seg[k + n - 1];
    }
    void advance() {        /*iが複数回更新されている場合最後だけみればいいが         * そのような使い方はしないため前からみる*/        while (!q.empty()) {
            int i;
            M v;
            tie(i, v) = q.front();
            q.pop();
            if (seg[i + n - 1] != v) { update(i, seg[i + n - 1]); }
        }
    }
    int most_r(int r, M v) {
        M d = get(0, r, false);
        if (get(0, r, false) > v) {
            err("most_r : ", r, ")  <= ", v, " :  ", -1);
            debu();
            return -1;
        }
        int l = 0;
        while (r - l > 1) {
            int m = (l + r) >> 1;
            if (get(m, r, false) <= v) { l = m; } else { r = m; }
        }
        err("most_r : ", r, ")  <= ", v, " :  ", l);
        debu();
        return l;
    }
    //r)でv以下を満たす最右のindexを返す
    int most_r(M v) { return most_r(n, v); }
    //[lで v以下を満たす最左のindexを返す
    int most_l(int l, M v) {
        if (get(l, n, false) > v) {
            err("most_l : ", "[", l, "<= ", v, " :  ", -1);
            debu();
            return -1;
        }
        int r = n;
        while (r - l > 1) {
            int m = (l + r) >> 1;
            if (get(l, m, false) <= v) { r = m; } else { l = m; }
        }
        err("most_l : ", "[", l, "<= ", v, " :  ", l);
        debu();
        return l;
    }
    int most_l(M v) { return most_l(0, v); }
#ifdef _DEBUG
    void debu() {
        rep(i, min(n, 10ll)) {
            int v = seg[i + n - 1];
            if (v == MIN(ll) || v == MAX(ll)) { cerr << "e "; } else { cerr << v << " "; }
        }
        cerr << "" << endl << endl;
    }
    void debup(int k, int v) {
        cerr << "upd : [" << k << "] = " << v << endl;
        debu();
    }
    void debge(int l, int r, M v) {
        cerr << "get : [" << l << ", " << r << ") = " << v << endl;
        debu();
    }
#else
    void debu(){;}    void debup(int k,int v){;}    void debge(int l,int r,M v){;}/*@formatter:on*/
#endif
};
/*@formatter:off*/

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
#endif
    return 0;
};

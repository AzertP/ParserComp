#include <bits/stdc++.h>

using namespace std;

#define int long long
// #define double long double
#define FOR(i, a, b) for(int i = (a); i < (b); ++i)
#define FORR(i, a, b) for(int i = (a); i > (b); --i)
#define REP(i, n) for(int i = 0; i < (n); ++i)
#define REPR(i, n) for(int i = n; i >= 0; i--)
#define FOREACH(x, a) for(auto &(x) : (a))
#define VECCIN(x)                                                              \
    for(auto &youso_ : (x)) cin >> youso_
#define bitcnt(x) __builtin_popcount(x)
#define lbit(x) __builtin_ffsll(x)
#define rbit(x) (64 - __builtin_clzll(x))
#define fi first
#define se second
#define All(a) (a).begin(), (a).end()
#define rAll(a) (a).rbegin(), (a).rend()
#define sz(s) (ll) s.size()
#define cinfast() cin.tie(0), ios::sync_with_stdio(false)
#define debug(x) cerr << #x << " = " << (x) << endl;
#define loop(x)                                                                \
    LCIN(loop);                                                                \
    while(loop--) {                                                            \
        x;                                                                     \
    }
#define PERM(c)                                                                \
    sort(All(c));                                                              \
    for(bool cp = true; cp; cp = next_permutation(All(c)))
#define COMB(n, k)                                                             \
    for(ll bit = (1LL << k) - 1; bit < (1LL << n); bit = next_combination(bit))
#define PERM2(n, k)                                                            \
    COMB(n, k) {                                                               \
        vector<int> sel;                                                       \
        for(int bitindex = 0; bitindex < n; bitindex++)                        \
            if(bit >> bitindex & 1) sel.emplace_back(bitindex);                \
        PERM(sel) { Printv(sel); }                                             \
    }
#define MKORDER(n)                                                             \
    vector<int> od(n);                                                         \
    iota(All(od), 0LL);

template <typename T = long long> inline T IN() {
    T x;
    cin >> x;
    return (x);
}
inline void CIN() {}
template <class Head, class... Tail>
inline void CIN(Head &&head, Tail &&... tail) {
    cin >> head;
    CIN(move(tail)...);
}
template <class Head> inline void COUT(Head &&head) { cout << (head) << "\n"; }
template <class Head, class... Tail>
inline void COUT(Head &&head, Tail &&... tail) {
    cout << (head) << " ";
    COUT(forward<Tail>(tail)...);
}

#define CCIN(...)                                                              \
    char __VA_ARGS__;                                                          \
    CIN(__VA_ARGS__)
#define DCIN(...)                                                              \
    double __VA_ARGS__;                                                        \
    CIN(__VA_ARGS__)
#define LCIN(...)                                                              \
    long long __VA_ARGS__;                                                     \
    CIN(__VA_ARGS__)
#define SCIN(...)                                                              \
    string __VA_ARGS__;                                                        \
    CIN(__VA_ARGS__)
#define Printv(v)                                                              \
    {                                                                          \
        REP(hoge, v.size())                                                    \
        cout << v[hoge] << (hoge == v.size() - 1 ? "" : " ");                  \
        cout << "\n";                                                          \
    }
template <typename T = string> inline void eputs(T s) {
    cout << s << "\n";
    exit(0);
}
template <typename A, size_t N, typename T>
void Fill(A (&array)[N], const T &val) {
    std::fill((T *)array, (T *)(array + N), val);
}

long long next_combination(long long sub) {
    long long x = sub & -sub, y = sub + x;
    return (((sub & ~y) / x) >> 1) | y;
}
template <class T> inline bool chmax(T &a, const T &b) {
    if(a < b) {
        a = b;
        return 1;
    }
    return 0;
}
template <class T> inline bool chmin(T &a, const T &b) {
    if(a > b) {
        a = b;
        return 1;
    }
    return 0;
}

// generic lambdas
template <typename first>
#if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard)
[[nodiscard]]
#elif defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ >= 4)
__attribute__((warn_unused_result))
#endif // defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard)
static inline constexpr decltype(auto)
fix(first &&f) noexcept {
    return [f = std::forward<first>(f)](auto &&... args) {
        return f(f, std::forward<decltype(args)>(args)...);
    };
}

template <typename T> using PQG = priority_queue<T, vector<T>, greater<T>>;
template <typename T> using PQ = priority_queue<T>;

typedef long long ll;
typedef vector<ll> VL;
typedef vector<VL> VVL;
typedef pair<ll, ll> PL;
typedef vector<PL> VPL;
typedef vector<bool> VB;
typedef vector<double> VD;
typedef vector<string> VS;

const int INF = 1e9;
const ll LINF = 1e18;
const int MOD = 1e9 + 7;
// const int MOD = 998244353;
const ll dw[] = {1, 1, 0, -1, -1, -1, 0, 1};
const ll dh[] = {0, 1, 1, 1, 0, -1, -1, -1};
#define PI 3.141592653589793230
#define EPS 1e-7

// 1000000007 で割ったあまりを扱う構造体
class mint {
  public:
    static long long MOD;
    long long val;
    mint(long long v = 0) noexcept : val(v % MOD) {
        if(val < 0) v += MOD;
    }
    int getmod() { return MOD; }
    mint operator-() const noexcept { return val ? MOD - val : 0; }
    mint operator+(const mint &r) const noexcept { return mint(*this) += r; }
    mint operator-(const mint &r) const noexcept { return mint(*this) -= r; }
    mint operator*(const mint &r) const noexcept { return mint(*this) *= r; }
    mint operator/(const mint &r) const noexcept { return mint(*this) /= r; }
    mint &operator+=(const mint &r) noexcept {
        val += r.val;
        val = (val + MOD) % MOD;
        return *this;
    }
    mint &operator-=(const mint &r) noexcept {
        val -= r.val;
        val = (val + MOD) % MOD;
        return *this;
    }
    mint &operator*=(const mint &r) noexcept {
        val = val * r.val % MOD;
        return *this;
    }
    mint &operator/=(const mint &r) noexcept {
        long long a = r.val, b = MOD, u = 1, v = 0;
        while(b) {
            long long t = a / b;
            a -= t * b;
            swap(a, b);
            u -= t * v;
            swap(u, v);
        }
        val = val * u % MOD;
        if(val < 0) val += MOD;
        return *this;
    }
    bool operator==(const mint &r) const noexcept { return this->val == r.val; }
    bool operator!=(const mint &r) const noexcept { return this->val != r.val; }
    friend ostream &operator<<(ostream &os, const mint &x) noexcept {
        return os << x.val;
    }
    friend istream &operator>>(istream &is, mint &x) noexcept {
        return is >> x.val;
    }
    friend mint modpow(const mint &a, long long n) noexcept {
        if(n == 0) return 1;
        mint t = modpow(a, n / 2);
        t = t * t;
        if(n & 1) t = t * a;
        return t;
    }
};
ll mint::MOD = 1e9 + 7;
ll const MAX_SIZE = 1e6 + 1;
mint inv[MAX_SIZE];
mint fact[MAX_SIZE];
mint factinv[MAX_SIZE];
void cominit() {
    inv[1] = 1;
    for(int i = 2; i < MAX_SIZE; i++) {
        inv[i] = (-inv[mint::MOD % i]) * (mint::MOD / i);
    }
    fact[0] = factinv[0] = 1;
    for(int i = 1; i < MAX_SIZE; i++) {
        fact[i] = mint(i) * fact[i - 1];
        factinv[i] = inv[i] * factinv[i - 1];
    }
}
mint comb(int n, int k) {
    if(n >= 0 && k >= 0 && n - k >= 0) {
        return fact[n] * factinv[k] * factinv[n - k];
    }
    return 0;
}
mint hcomb(ll n, ll k) {
    if(n == 0 && k == 0) return 1;
    return comb(n + k - 1, k);
}
ll gcd(ll x, ll y) { return y ? gcd(y, x % y) : x; }

VL divisor(ll n) {
    VL res;
    for(ll i = 1; i * i <= n; i++) {
        if(n % i == 0) {
            res.push_back(i);
            if(i != n / i) res.push_back(n / i);
        }
    }
    return res;
}
void solve() {
    LCIN(N, K);
    mint ans = 0;
    vector<mint> cnt(K + 1);
    FORR(i, K, 0) {
        cnt[i] += modpow((mint)(K / i), N);
        ans += cnt[i] * i;
        FOREACH(d, divisor(i)) {
            if(d == i) continue;
            cnt[d] -= cnt[i];
        }
    }
    COUT(ans);
}

signed main() {
    cinfast();
    // cout << fixed << setprecision(12);
    // loop(solve());
    solve();
}

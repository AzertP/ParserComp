#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <numeric>
#include <cmath>
#include <limits>
#include <bitset> 
#include <iterator> 
#include <map>
#include <unordered_map>
#include <queue>
#include <set>
#include <deque>
#include <regex>
#include <list>
#include <unordered_set>
#include <assert.h>
#include <functional>
//#include <intrin.h>
#define rep(i,n) for(ll i=0;i<(n); ++i)
#define rep1(i,n) for(ll i=1;i<=(n); ++i)
#define per(i,n) for(ll i=n-1;i>=0; --i)
#define All(x) (x).begin(),(x).end()
#define rAll(x) (x).rbegin(),(x).rend()
#define _GLIBCXX_DEBUG
using namespace std;

//省略表現
const long long INF = 1LL << 60;
using ll = long long;
using P = pair<ll, ll>;
using GraphEdge = vector <vector<P>>;
using Graph = vector <vector<ll>>;
const int mod = 1000000007;
//実行時間高速化
struct __INIT { __INIT() { cin.tie(nullptr); ios::sync_with_stdio(false); } } __init;
//辺にコストを持たせる構造体
/**
struct Edge{
    ll to;
    ll cost;
    Edge(ll t, ll c) : to(t), cost(c) {}
};
*/
//unorderd_mapのキーにpairを使えるようにするための構造体
struct HashPair {

    //注意 constがいる
    template<class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {

        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);

        size_t seed = 0;
        seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
//最大値、最小値用のdp用の関数
template<class T> inline bool chmax(T& a, T b) { if (a < b) { a = b; return 1; } return 0; }
template<class T> inline bool chmin(T& a, T b) { if (a > b) { a = b; return 1; } return 0; }
//固定長配列の大きさ（dp用）
template<typename TYPE, size_t SIZE>
size_t array_length(const TYPE(&)[SIZE])
{
    return SIZE;
}
//'0'~'9を数値に変換する
ll stochr(char c) {
    return c - '0';
}
//剰余計算
struct mint {
    ll x; // typedef long long ll;
    mint(ll x = 0) :x((x% mod + mod) % mod) {}
    mint& operator+=(const mint a) {
        if ((x += a.x) >= mod) x -= mod;
        return *this;
    }
    mint& operator-=(const mint a) {
        if ((x += mod - a.x) >= mod) x -= mod;
        return *this;
    }
    mint& operator*=(const mint a) {
        (x *= a.x) %= mod;
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
    // for prime mod
    mint inv() const {
        return pow(mod - 2);
    }
    mint& operator/=(const mint a) {
        return (*this) *= a.inv();
    }
    mint operator/(const mint a) const {
        mint res(*this);
        return res /= a;
    }
};
//組み合わせ(剰余付き)
//combination c(n)で初期化する必要がある。（nはCの左側の数）
struct combination {
    vector<mint> fact, ifact;
    combination(int n) :fact(n + 1), ifact(n + 1) {
        assert(n < mod);
        fact[0] = 1;
        for (int i = 1; i <= n; ++i) fact[i] = fact[i - 1] * i;
        ifact[n] = fact[n].inv();
        for (int i = n; i >= 1; --i) ifact[i - 1] = ifact[i] * i;
    }
    mint operator()(int n, int k) {
        if (k < 0 || k > n) return 0;
        return fact[n] * ifact[k] * ifact[n - k];
    }
};
//桁取得
ll getDigit(ll n) {
    ll cnt = 0;
    while (n != 0) {
        n /= 10;
        cnt++;
    }
    return cnt;
}
//素数判定（O（sqrt(n)））
//繰り返す数がn/sqrt(n)回以下なら、こっちの素数判定の方が速いはず（k*sqrt(n)<=n）
bool isPrime(ll n) {
    if (n == 1)return false;
    for (ll i = 2; i * i <= n; i++) {
        if (n % i == 0)return false;
    }
    return true;
}
//素因数分解（O(logN)）
//素因数分解したい値が10^8以上ならこっちを使う
vector<pair<ll, ll>> factorize(ll n) {
    vector<pair<ll, ll>> res;
    for (ll i = 2; i * i <= n; ++i) {
        if (n % i) continue;
        else {
            res.emplace_back(i, 0);
            while (n % i == 0) {
                n /= i;
                res.back().second++;
            }
        }
    }
    if (n != 1) res.emplace_back(n, 1);
    return res;
}
//エラストテネスの篩（O(NloglogN))
struct Sieve {
    ll n;
    vector <ll> f, primes;
    Sieve(ll n = 1) :n(n), f(n + 1) {
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

    //素数判定
    bool isPrime(ll x) { return f[x] == x; }
    vector<ll> factorList(ll x) {
        vector<ll> res;
        while (x != 1) {
            res.push_back(f[x]);
            x /= f[x];
        }
        return res;
    }

    //素因数分解
    vector<P> factor(ll x) {
        vector<ll> fl = factorList(x);
        if (fl.size() == 0) return {};
        vector<P> res(1, P(fl[0], 0));
        for (ll p : fl) {
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
ll gcd(ll a, ll b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}
//ペア比較順序
typedef pair<ll, ll> pair_t;
bool PSecComp(const pair_t& a, const pair_t& b) {
    if (a.second == b.second)return  a.first > b.first;
    else {
        return a.second > b.second;
    }
}

int main() {
    string s, t;
    cin >> s >> t;
    unordered_map<char, char>mpch;
    for (char i = 'a'; i <= 'z'; i++)mpch[i] = i;
    for (ll i = 0; i < s.length(); i++) {
        if (s[i] != t[i]) {
            for (auto x : mpch) {
                if (x.second == t[i]) {
                    x.second = s[i];
                }
            }
            mpch[s[i]] = t[i];
        }
    }
    unordered_map<char, ll>mpbu;
    for (auto x : mpch) {
        mpbu[x.second]++;
    }
    for (ll i = 0; i < s.length(); i++) {
        if (mpch[s[i]] != t[i] || mpbu[s[i]] >= 2)return cout << "No" << endl,0;
    }
    cout << "Yes" << endl;
}

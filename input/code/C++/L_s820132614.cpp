#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(),(x).end()
#define ll long long
#define ld long double
#define vl vector<long long>
#define vvl vector<vector<long long>>
#define rep(i, n) for (ll i = 0; i < (ll)(n); i++)
#define rep2(i, s, n) for (ll i = (s); i < (ll)(n); i++)
#define rrep(i, x) for (ll i = ((ll)(x)-1); i >= 0; i--)
#define pll pair<long long,long long>
#define pb push_back
#define mp make_pair
#define mt make_tuple
#define vvc vector<vector<char>>
#define vc vector<char>
#define vvb vector<vector<bool>>
#define vb vector<bool>
#define elif else if
#define maxe(x) *max_element(all(x))
#define mine(x) *min_element(all(x))
#define Size(x) ((int)(x).size())
const long long INF = 1LL << 60;
const long double pi = 3.1415926535897932;
long long MOD = 1000000007;
int dx[4]={1,0,-1,0};
int dy[4]={0,1,0,-1};

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

    vector<long long> divisor(long long n) {
        vector<long long> ret;
        for (long long i = 1; i * i <= n; i++) {
            if (n % i == 0) {
                ret.push_back(i);
                if (i * i != n) ret.push_back(n / i);
            }
        }
        sort(ret.begin(), ret.end()); // 昇順に並べる
        return ret;
    }

    map< ll, ll > prime_factor(ll n) {
        map< ll, ll > ret;
        for(ll i = 2; i * i <= n; i++) {
            while(n % i == 0) {
                ret[i]++;
                n /= i;
            }
        }
        if(n != 1) ret[n] = 1;
        return ret;
    }

    //(mod m)でのaの逆元を計算する
    //a/b(mod m)=a(mod m)*modinv(b,m)
    long long modinv(long long a, long long m) {
        long long b = m, u = 1, v = 0;
        while (b) {
            long long t = a / b;
            a -= t * b; swap(a, b);
            u -= t * v; swap(u, v);
        }
        u %= m;
        if (u < 0) u += m;
        return u;
    }

    long long modpow(long long a, long long n, long long mod) {
        long long res = 1;
        while (n > 0) {
            if (n & 1) res = res * a % mod;
            a = a * a % mod;
            n >>= 1;
        }
        return res;
    }

    ll modnCr(ll n, ll r, ll mod) {
        ll res = 1;
        rep (i, r) {
          res = res * (n - i) % mod * modpow(i + 1, mod - 2, mod) % mod;
        }
        return res;
      }

    struct UnionFind {
        vector<int> par;
    
        UnionFind(int n) : par(n, -1) { }
    
        int root(int x) {
            if (par[x] < 0) return x;
            else return par[x] = root(par[x]);
        }
    
        bool issame(int x, int y) {
            return root(x) == root(y);
        }
    
        bool merge(int x, int y) {
            x = root(x); y = root(y);
            if (x == y) return false;
            if (par[x] > par[y]) swap(x, y); // merge technique
            par[x] += par[y];
            par[y] = x;
            return true;
        }
    
        int size(int x) {
            return -par[root(x)];
        }
    };
//mint,----------------------------------------------------
    signed main(){
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
        ll m,f,b;
        cin>>m>>f>>b;
        if(m>=b)cout<<0<<endl;
        elif(m+f>=b)cout<<b-m<<endl;
        else cout<<"NA"<<endl;
    }

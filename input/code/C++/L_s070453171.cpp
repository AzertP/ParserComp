#pragma target("avx")
#include <bits/stdc++.h>

template <std::uint_fast64_t Modulus>
class modint {
  using u64 = std::uint_fast64_t;

 public:
  u64 a;

  constexpr modint(const u64 x = 0) noexcept : a(x % Modulus) {}
  constexpr u64 &value() noexcept { return a; }
  constexpr const u64 &value() const noexcept { return a; }
  constexpr modint operator+(const modint rhs) const noexcept {
    return modint(*this) += rhs;
  }
  constexpr modint operator-(const modint rhs) const noexcept {
    return modint(*this) -= rhs;
  }
  constexpr modint operator*(const modint rhs) const noexcept {
    return modint(*this) *= rhs;
  }
  constexpr modint operator/(const modint rhs) const noexcept {
    return modint(*this) /= rhs;
  }
  constexpr modint &operator+=(const modint rhs) noexcept {
    a += rhs.a;
    if (a >= Modulus) {
      a -= Modulus;
    }
    return *this;
  }
  constexpr modint &operator-=(const modint rhs) noexcept {
    if (a < rhs.a) {
      a += Modulus;
    }
    a -= rhs.a;
    return *this;
  }
  constexpr modint &operator*=(const modint rhs) noexcept {
    a = a * rhs.a % Modulus;
    return *this;
  }
  constexpr modint &operator/=(modint rhs) noexcept {
    u64 exp = Modulus - 2;
    while (exp) {
      if (exp % 2) {
        *this *= rhs;
      }
      rhs *= rhs;
      exp /= 2;
    }
    return *this;
  }
};

using namespace std;

typedef long long ll;
typedef long double ld;
typedef pair<ll, ll> P;
typedef vector<ll> V;
typedef unordered_map<ll, ll> U_MAP;
typedef priority_queue<ll> pq;
typedef priority_queue<ll, vector<ll>, greater<ll>> rpq;
const int INF = 1e9, MOD = 1e9 + 7, ohara = 1e6 + 10;
const ll LINF = 1e18;

#define rep(i, n) for (ll(i) = 0; (i) < (int)(n); (i)++)
#define rrep(i, a, b) for (ll i = (a); i < (b); i++)
#define rrrep(i, a, b) for (ll i = (a); i >= (b); i--)
#define all(v) (v).begin(), (v).end()
#define Size(n) (n).size()
#define Cout(x) cout << (x) << endl
#define doublecout(a) cout << fixed << setprecision(15) << a << endl;
#define fi first
#define se second
#define m_p make_pair
#define p_b push_back
string to_string(string s) { return '"' + s + '"'; }
string to_string(const char *s) { return to_string((string)s); }
string to_string(bool b) { return (b ? "true" : "false"); }
template <typename A, typename B>
string to_string(pair<A, B> p) {
  return "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
}
template <typename A>
string to_string(A v) {
  bool first = true;
  string res = "{";
  for (const auto &x : v) {
    if (!first) {
      res += ", ";
    }
    first = false;
    res += to_string(x);
  }
  res += "}";
  return res;
}
void debug_out() { cerr << endl; }
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) {
  cerr << " " << to_string(H);
  debug_out(T...);
}
#define debug(...) cerr << "[" << #__VA_ARGS__ << "]:", debug_out(__VA_ARGS__)

//------ Believe yourself as a genius!!!!!! ------

int dy[] = {1, 0, -1, 0};
int dx[] = {0, 1, 0, -1};
// int dy[]={-1,0,1,-1,1,-1,0,1};int dx[]={-1,-1,-1,0,0,1,1,1};
string alph("abcdefghijklmnopqrstuvwxyz"), s;
ll n, cnt, a, b, c, d, tmp, m, h, w, x, y, sum, k, q;
ll f[3 * ohara + 100], fr[3 * ohara + 100];

ll mod_pow(ll xx, ll nn, ll mod) {
  ll ret = 1;
  while (nn > 0) {
    if (nn & 1) ret = ret * xx % mod;
    xx = xx * xx % mod;
    nn >>= 1;
  }
  return ret;
}

ll ncr(ll nn, ll r) {
  if (nn < r) return 0;
  return f[nn] * fr[r] % MOD * fr[nn - r] % MOD;
}

int main(void) {
  cin.tie(0);
  cout.tie(0);
  ios::sync_with_stdio(false);

  using mint = modint<MOD>;

  cin >> a >> b >> c >> d;
  mint ans;

  n = 3 * ohara;
  f[0] = 1;
  rrep(i, 1, n + 3) { f[i] = f[i - 1] * i % MOD; }
  fr[n + 2] = mod_pow(f[n + 2], MOD - 2, MOD);
  rrrep(i, n + 1, 0) { fr[i] = fr[i + 1] * (i + 1) % MOD; }

  x = c, y = d;
  ans = ncr(x + y + 2, y + 2);
  ans /= mint(x + 1);
  ans *= mint(y + 2);
  ans -= mint(x + 1) / mint(x + 1);

  mint remove1, remove2, add;

  x = a - 1, y = d;
  remove1 = ncr(x + y + 2, y + 2);
  remove1 /= mint(x + 1);
  remove1 *= mint(y + 2);
  remove1 -= mint(x + 1) / mint(x + 1);

  x = c, y = b - 1;
  remove2 = ncr(x + y + 2, y + 2);
  remove2 /= mint(x + 1);
  remove2 *= mint(y + 2);
  remove2 -= mint(x + 1) / mint(x + 1);

  x = a - 1, y = b - 1;
  add = ncr(x + y + 2, y + 2);
  add /= mint(x + 1);
  add *= mint(y + 2);
  add -= mint(x + 1) / mint(x + 1);

  ans = ans - remove1 - remove2 + add;
  Cout(ans.value());
  return 0;
}

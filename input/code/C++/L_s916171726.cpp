#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <regex>
#include <set>
#include <stack>
#include <string>
#include <vector>

const int MOD = 1e9 + 7;
const int iINF = 1000000000;
const long long int llINF = 1000000000000000000;

using namespace std;
using ll = long long int;
using vl = vector<ll>;
using vvl = vector<vector<ll>>;
using vvvl = vector<vector<vector<ll>>>;

typedef pair<ll, ll> pll;
bool paircomp(const pll &a, const pll &b) {
  if (a.first == b.first)
    return a.second < b.second;
  return a.first < b.first;
}
struct multi {
  ll first;
  ll second;
  ll third;
};
bool multicomp(const multi &a, const multi &b) {
  if (a.first == b.first)
    return a.second < b.second;
  return a.first < b.first;
}
#define REP(i, n) for (ll i = 0; i < (n); i++)
#define RREP(i, n) for (ll i = (n)-1; i >= 0; i--)
#define FOR(i, a, b) for (ll i = (a); i < (b); i++)
#define AUTO(i, m) for (auto &i : m)
#define ALL(a) (a).begin(), (a).end()
#define MAX(vec) *std::max_element(vec.begin(), vec.end())
#define MIN(vec) *std::min_element(vec.begin(), vec.end())
#define ARGMAX(vec)                                                            \
  std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()))
#define ARGMIN(vec)                                                            \
  std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()))
#define REV(T) greater<T>()
#define PQ(T) priority_queue<T, vector<T>, greater<T>>
#define VVL(a, b, c) vector<vector<ll>>(a, vector<ll>(b, c))
#define VVVL(a, b, c, d)                                                       \
  vector<vector<vector<ll>>>(a, vector<vector<ll>>(b, vector<ll>(c, d)))
#define SP(a) setprecision(a)
#define SQRT(a) sqrt((long double)(a))
#define DPOW(a, b) pow((long double)(a), (long double)(b))
#define UNIQUE(vec)                                                            \
  do {                                                                         \
    sort(ALL((vec)));                                                          \
    (vec).erase(std::unique(ALL((vec))), (vec).end());                         \
  } while (0)

ll POW(ll n, ll m) {
  if (m == 0) {
    return 1;
  } else if (m % 2 == 0) {
    ll tmp = POW(n, m / 2);
    return (tmp * tmp);
  } else {
    return (n * POW(n, m - 1));
  }
}

int dx[4] = {1, 0, -1, 0};
int dy[4] = {0, 1, 0, -1};

ll func(vvl &S, ll &h, ll &w, ll y, ll x, ll score) {
  if (y < 0 || y >= h || x < 0 || x >= w)
    return llINF;
  if (S[y][x] == 3)
    return score;
  if (score == 10)
    return llINF;
  vl result;
  if (y <= h - 2 && S[y + 1][x] != 1) {
    ll yc = y;
    while (yc < h - 1 && S[yc + 1][x] != 1) {
      yc++;
      if (S[yc][x] == 3)
        return score + 1;
    }
    if (yc <= h - 2 && S[yc + 1][x] == 1)
      S[yc + 1][x] = 0;
    else if (S[yc][x] != 3)
      yc = h;
    result.push_back(func(S, h, w, yc, x, score + 1));
    if (yc <= h - 2 && S[yc + 1][x] == 0)
      S[yc + 1][x] = 1;
  }
  if (y > 0 && S[y - 1][x] != 1) {
    ll yc = y;
    while (yc > 0 && S[yc - 1][x] != 1) {
      yc--;
      if (S[yc][x] == 3)
        return score + 1;
    }
    if (yc > 0 && S[yc - 1][x] == 1)
      S[yc - 1][x] = 0;
    else if (S[yc][x] != 3)
      yc = -1;
    result.push_back(func(S, h, w, yc, x, score + 1));
    if (yc > 0 && S[yc - 1][x] == 0)
      S[yc - 1][x] = 1;
  }
  if (x <= w - 2 && S[y][x + 1] != 1) {
    ll xc = x;
    while (xc < w - 1 && S[y][xc + 1] != 1) {
      xc++;
      if (S[y][xc] == 3)
        return score + 1;
    }
    if (xc <= w - 2 && S[y][xc + 1] == 1)
      S[y][xc + 1] = 0;
    else if (S[y][xc] != 3)
      xc = w;
    result.push_back(func(S, h, w, y, xc, score + 1));
    if (xc <= w - 2 && S[y][xc + 1] == 0)
      S[y][xc + 1] = 1;
  }
  if (x > 0 && S[y][x - 1] != 1) {
    ll xc = x;
    while (xc > 0 && S[y][xc - 1] != 1) {
      xc--;
      if (S[y][xc] == 3)
        return score + 1;
    }
    if (xc > 0 && S[y][xc - 1] == 1)
      S[y][xc - 1] = 0;
    else if (S[y][xc] != 3)
      xc = -1;
    result.push_back(func(S, h, w, y, xc, score + 1));
    if (xc > 0 && S[y][xc - 1] == 0)
      S[y][xc - 1] = 1;
  }
  if (result.size() == 0)
    return llINF;
  else
    return MIN(result);
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  vl ans;
  while (true) {
    ll w = 0, h = 0, x = -1, y = -1;
    cin >> w >> h;
    if (w == 0 || h == 0)
      break;
    vvl S = VVL(h + 1, w + 1, 0);

    REP(i, h) REP(j, w) {
      cin >> S[i][j];
      if (S[i][j] == 2) {
        y = i;
        x = j;
        S[i][j] = 0;
      }
    }
    ans.push_back(func(S, h, w, y, x, 0));
  }
  AUTO(a, ans) {
    if (a == llINF)
      cout << -1 << endl;
    else
      cout << a << endl;
  }
  return 0;
}


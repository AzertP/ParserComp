#include <bits/stdc++.h>
#define rep(i, n) for (int i = 0; i < (n); ++i)
using namespace std;
using ll = long long;
using P = pair<int, int>;

const ll INF = ll(1e18) + 1;

int main() {
  int n;
  ll k;
  cin >> n >> k;
  vector<ll> a(n);
  rep(i, n) cin >> a[i];
  sort(a.begin(), a.end());
  ll l = -INF, r = INF;
  while (l + 1 < r) {
    ll x = (l + r) / 2;
    bool ok = [&] {
      ll tot = 0;
      rep(i, n) {
        if (a[i] < 0) {
          int l = -1, r = n;
          while (l + 1 < r) {
            int c = (l + r) / 2;
            if (a[c] * a[i] < x)
              r = c;
            else
              l = c;
          }
          tot += n - r;
        } else {
          int l = -1;
          int r = n;
          while (l + 1 < r) {
            int c = (l + r) / 2;
            if (a[c] * a[i] < x)
              l = c;
            else
              r = c;
          }
          tot += r;
        }
        if (a[i] * a[i] < x) tot--;
      }
      tot /= 2;
      return tot < k;
    }();
    if (ok)
      l = x;
    else
      r = x;
  }
  cout << l << endl;
  return 0;
}
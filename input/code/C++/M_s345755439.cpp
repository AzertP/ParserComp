#include <iostream>

using namespace std;

#ifndef ONLINE_JUDGE
  #include <fstream>
  #define cin fin
  #define cout fout
  ifstream fin ("test.in");
  ofstream fout ("test.out");
#endif

int p[10];
int n, m, q;
int a[60], b[60], c[60], d[60];

int f (int i, int fn) {
  if (i == n) {
    int ans = 0;
    for (int j = 0; j < q; j++) {
      if (p[b[j]] - p[a[j]] == c[j])
        ans += d[j];
    }
    return ans;
  }
  int ans = 0;
  for (int j = fn; j <= m; j++) {
    p[i] = j;
    ans = max (ans, f(i + 1, j));
  }
  return ans;
}

int main() {
  cin >> n >> m >> q;
  for (int i = 0; i < q; i++)
    cin >> a[i] >> b[i] >> c[i] >> d[i];
  for (int i = 0; i < q; i++) {
    a[i]--; b[i]--;
  }
  cout << f (0, 1) << endl;
  return 0;
}

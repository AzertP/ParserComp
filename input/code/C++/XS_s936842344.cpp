#include <bits/stdc++.h>

using namespace std;
int main() {
  int a, b;
  cin >> a >> b;
  int r = a - b * 2;
  cout << (r < 0 ? 0 : r) << endl;
  return 0;
}
#include <bits/stdc++.h>

using namespace std;

bool solve(const std::vector<int>& as) { return true; }

int main() {
  int n;
  std::cin >> n;
  int odd = 0;
  int a;
  while (std::cin >> a) {
    odd += a % 2;
  }
  std::cout << (odd % 2 ? "NO" : "YES") << std::endl;
  return 0;
}
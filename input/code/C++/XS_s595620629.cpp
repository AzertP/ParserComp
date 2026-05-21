#include <bits/stdc++.h>
using namespace std;
#define rep(i, n) for (int i = 0; i < (n); i++)
int main(void)
{
  float a, b, area, perimeter;
  cin >> a >> b;
  area = a * b;
  perimeter = a* 2 + b * 2;
  cout << area << " " << perimeter << endl;
}

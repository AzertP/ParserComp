#include <bits/stdc++.h>

using namespace std;
using ll = long long;

int main()
{
    ll a, b;
    cin >> a >> b;
    ll ans =(a + b);
    if (ans % 2 == 0){
        cout << ans / 2;
    } else {
        cout << "IMPOSSIBLE";
    }
    return 0;
}

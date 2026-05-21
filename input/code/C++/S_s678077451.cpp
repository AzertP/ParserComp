#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main()
{
    int n, k, q, tmp;
    cin >> n >> k >> q;

    vector<ll> a(n, k - q);

    for(int i = 0; i < q; ++i)
    {
        cin >> tmp;
        a[tmp - 1]++;
    }

    for(int i = 0;  i < n; ++i)
        cout << (a[i] > 0 ? "Yes" : "No") << "\n";
    return 0;
}
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i,n) for (int i = 0; i < (n); i++)

typedef pair<int, int> P;

int main() {

    int n;
    cin >> n;
    vector<int> b(n);
    rep(i,n-1) cin >> b[i];
    ll ans = b[0];
    rep(i,n-1){
        ans += min(b[i],b[i+1]);
    }
    ans += b[n-2];
    cout << ans << endl;

    return 0;
}
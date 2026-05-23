// AtCoder template
using namespace std;
typedef long long ll;

int main(){
    cin.tie(0);
    ios::sync_with_stdio(false);

    ll n, ans = 0LL; cin >> n;
    vector<ll> f(n+1, 0LL);
    for(ll i = 1LL; i <= n; ++i) for(ll l = 1LL; l*i <= n; ++l) ++f[l*i];
    for(ll i = 1LL; i <= n; ++i) ans += i*f[i];
    cout << ans << endl;
}

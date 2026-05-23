using namespace std;
ll vs[110][2];
ll dp[110][100100];

void solve() {
    int N, W;
    cin >> N >> W;
    for (ll i = 1; i <= N; ++i) cin >> vs[i][0] >> vs[i][1];
    for (ll i = 0; i <= W; ++i) dp[0][i] = 0;
    for (ll i = 0; i <= N; ++i) dp[i][0] = 0;

    for (ll i = 1; i <= N; ++i) {
        ll* vi = vs[i];
        for (ll w = 1; w <= W; ++w) {
            if (w < vi[0]) dp[i][w] = dp[i - 1][w];
            else dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - vi[0]] + vi[1]);
        }
    }
    cout << dp[N][W] << endl;
}

int main() {
    solve();
    return 0;
}

#pragma region Macros
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using Graph = vector<vector<int>>;
using WGraph = vector<vector<pair<int, ll>>>;

template<class T>bool chmax(T &a, const T &b) { if (b > a) { a = b; return true; } return false; }
template<class T>bool chmin(T &a, const T &b) { if (b < a) { a = b; return true; } return false; }

constexpr int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
constexpr int INF = 1e9;
constexpr long long LINF = 1e18;
constexpr long long MOD = 1e9 + 7;
constexpr double EPS = 1e-10;
#pragma endregion

void solve() {
    int n;
    cin >> n;
    string s, pre1, pre2;
    cout << 0 << endl;
    cin >> s;
    if (s == "Vacant") return;
    pre1 = s;
    cout << n-1 << endl;
    cin >> s;
    if (s == "Vacant") return;
    pre2 = s;
    int a = 0, b = n-1;
    while (s != "Vacant") {
        int mid = (a + b) / 2;
        cout << mid << endl;
        cin >> s;
        if ((mid - a)%2 == 0 && s != pre1) {
            b = mid;
            pre2 = s;
        } else if ((mid - a)%2 != 0 && s == pre1) {
            b = mid;
            pre2 = s;
        } else if ((b - mid)%2 == 0 && s != pre2) {
            a = mid;
            pre1 = s;
        } else if ((b - mid)%2 != 0 && s == pre2) {
            a = mid;
            pre1 = s;
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout << fixed << setprecision(20);

    solve();

    return 0;
}

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int MOD = 1e9 + 7;
const int iinf = 1 << 28;
const long long llinf = 1ll << 60;
const double PI = 3.14159265;

int N;
vector<int> small;
vector<int> big;
vector<int> curr(8);
vector<vector<int>> perms(50000);
int idx;

int solve(int pos, bool restrict) {
    if (pos == N) {
        for (int i = 0; i < N; ++i) {
            if (curr[i] != big[i])
                return 1;
        }
        return 0;
    }

    int ans = 0;
    if (restrict) {
        for (int i = small[pos]; i <= big[pos]; ++i) {
            curr[pos] = i;
            if (i == big[pos])
                ans += solve(pos+1, true);
            else
                ans += solve(pos+1, false);
        }
    } else {
        for (int i = 1; i <= N; ++i) {
            curr[pos] = i;
            ans += solve(pos+1, false);
        }
    }
    return ans;
}

void get_perms(int pos, vector<bool> &used) {
    if (pos == N) {
        for (int i = 0; i < N; ++i) {
            perms[idx].push_back(curr[i]);
        }
        ++idx;
        return;
    }

    for (int i = 1; i <= N; ++i) {
        if (used[i]) continue;
        used[i] = true;
        curr[pos] = i;
        get_perms(pos+1, used);
        used[i] = false;
    }
}


int main() {
    cin >> N;
    for (int i = 0; i < N; ++i) {
        int a;
        cin >> a; small.push_back(a);
    }
    for (int i = 0; i < N; ++i) {
        int a;
        cin >> a; big.push_back(a);
    }

    for (int i = 0; i < N; ++i) {
        if (small[i] != big[i]) {
            if (small[i] > big[i]) {
                for (int j = 0; j < N; ++j) {
                    int tmp = small[j];
                    small[j] = big[j];
                    big[j] = tmp;
                }
            }
            break;
        }
    }

    /*
    for (int i = 0; i < N; ++i)
        cout << small[i] << "\t";
    cout << "\n";
    for (int i = 0; i < N; ++i)
        cout << big[i] << "\t";
    cout << endl;
    */

    vector<bool> used(10, false);
    get_perms(0, used);
    int ans = 0;
    for (int i = 0; i < idx; ++i) {
        bool ok = false;
        for (int j = 0; j < N; ++j) {
            if (perms[i][j] > small[j]) {
                ok = true;
                break;
            } else if (perms[i][j] < small[j]) {
                break;
            }
        }
        if (!ok) continue;
        ok = true;
        for (int j = 0; j < N; ++j) {
            if (perms[i][j] > big[j]) {
                ok = false;
                break;
            } else if (perms[i][j] < big[j]) {
                break;
            }
        }
        if (ok) ++ans;
    }

    cout << ans << endl;

    //cout << solve(0, true) << endl;


    return 0;
}

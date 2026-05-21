#define  _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#pragma comment (linker, "/STACK:526000000")

#include "bits/stdc++.h"

using namespace std;
typedef string::const_iterator State;
#define eps 1e-11L
#define MAX_MOD 1000000007LL
#define GYAKU 500000004LL

#define MOD 998244353LL
#define seg_size 262144 * 4LL
#define pb push_back
#define mp make_pair
typedef long long ll;
#define REP(a,b) for(long long (a) = 0;(a) < (b);++(a))
#define ALL(x) (x).begin(),(x).end()

void init() {
    iostream::sync_with_stdio(false);
    cout << fixed << setprecision(20);
}


#define int ll

unsigned long xor128() {
    static unsigned long x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned long t = (x ^ (x << 11));
    x = y; y = z; z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

int dp_max[300][300];
int dp_min[300][300];
int val[300];
int rest[300];
void solve() {
    string s;
    cin >> s;
    int now = 0;
    REP(i, s.length()) {
        if (s[i] == '(') {
            rest[now] |= 1;
        }
        else if (s[i] == ')') {
            rest[now - 1] |= 2;
        }
        else if (s[i] == '+') {
            val[now] = -1;
            now++;
        }
        else if (s[i] == '-') {
            val[now] = -2;
            now++;
        }
        else {
            val[now] = s[i] - '0';
            now++;
        }
    }
    REP(i, now + 1) {
        REP(q, now + 1) {
            dp_min[i][q] = 1e18;
            dp_max[i][q] = -1e18;
        }
    }
    REP(i, now) {
        if (val[i] >= 0) {
            dp_max[i][i] = val[i];
            dp_min[i][i] = val[i];
        }
    }
    for (int len = 3; len <= now; len += 2) {
        for (int q = 0; q < now - len + 1; ++q) {
            if (rest[q] == 2) continue;
            if (rest[q + len - 1] == 1) continue;
            for (int j = 1; j < len - 1; ++j) {
                if (val[q + j] == -1) {
                    dp_max[q][q + len - 1] = max(dp_max[q][q + len - 1], dp_max[q][q + j - 1] + dp_max[q + j + 1][q + len - 1]);
                    dp_min[q][q + len - 1] = min(dp_min[q][q + len - 1], dp_min[q][q + j - 1] + dp_min[q + j + 1][q + len - 1]);
                }
                if (val[q + j] == -2) {
                    dp_max[q][q + len - 1] = max(dp_max[q][q + len - 1], dp_max[q][q + j - 1] - dp_min[q + j + 1][q + len - 1]);
                    dp_min[q][q + len - 1] = min(dp_min[q][q + len - 1], dp_min[q][q + j - 1] - dp_max[q + j + 1][q + len - 1]);
                }
            }
        }
    }
    cout << dp_max[0][now - 1] << endl;
}

#undef int
int main() {
    init();
    solve();
}

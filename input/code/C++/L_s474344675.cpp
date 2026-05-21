#include <bits/stdc++.h>

#define rep(i, n) for(int i = 0; i<(n); i++)
#define chmax(x, y) x = max(x, y)
#define chmin(x, y) x = min(x, y)
using namespace std;
using ll = long long;

template<typename T>
struct Segtree {
    int n;
    T e;
    vector<T> dat;
    typedef function<T(T a, T b)> Func;
    Func f;

    Segtree() {}

    Segtree(int n_input, Func f_input, T e_input) {
        initialize(n_input, f_input, e_input);
    }

    void initialize(int n_input, Func f_input, T e_input) {
        f = f_input;
        e = e_input;
        n = 1;
        while (n < n_input) n <<= 1;
        dat.resize(2 * n - 1, e);
    }

    void update(int k, T a) {
        k += n - 1;
        dat[k] = a;
        while (k > 0) {
            k = (k - 1) / 2;
            dat[k] = f(dat[2 * k + 1], dat[2 * k + 2]);
        }
    }

    T get(int k) {
        return dat[k + n - 1];
    }

    T between(int a, int b) {
        return query(a, b + 1, 0, 0, n);
    }

    T query(int a, int b, int k, int l, int r) {
        if (r <= a || b <= l) return e;
        if (a <= l && r <= b) return dat[k];
        T vl = query(a, b, 2 * k + 1, l, (l + r) / 2);
        T vr = query(a, b, 2 * k + 2, (l + r) / 2, r);
        return f(vl, vr);
    }
};

using P = pair<int, int>;

int main() {
    int N;
    cin >> N;
    vector<P> LR(N);
    rep(i, N) {
        int l, r;
        cin >> l >> r;
        LR[i] = {l, r + 1};
    }
    sort(LR.begin(), LR.end());

    Segtree<int> Lmax(N, [](int a, int b) { return max(a, b); }, -1e9);
    Segtree<int> Rmin(N, [](int a, int b) { return min(a, b); }, 1e9 + 1);
    rep(i, N) {
        Lmax.update(i, LR[i].first);
        Rmin.update(i, LR[i].second);
    }

    int ans = 0;
    // 1区間孤立
    rep(i, N) {
        // i番目の区間だけを選ぶ
        int res1 = LR[i].second - LR[i].first;

        // i番目以外の区間の共通部分
        int l = max(Lmax.between(0, i - 1), Lmax.between(i + 1, N - 1));
        int r = min(Rmin.between(0, i - 1), Rmin.between(i + 1, N - 1));
        int res2 = max(0, r - l);
        ans = max(ans, res1 + res2);
    }

    // 左端ソートしてある境目で二分
    for (int i = 1; i < N; i++) {
        int res1 = Rmin.between(0, i - 1) - Lmax.between(0, i - 1);
        int res2 = Rmin.between(i, N - 1) - Lmax.between(i, N - 1);
        int res = max(0, res1) + max(0, res2);
        ans = max(ans, res);
    }

    cout << ans << endl;
}

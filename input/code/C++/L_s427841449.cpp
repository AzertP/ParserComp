#include <bits/stdc++.h>
using namespace std;

#define int long long
#define pb push_back
#define FOR(i,a,b) for (int (i) = (a); (i) < (b);(i)++)
#define rep(i,n) FOR (i,0,n)
#define vi vector<int>
#define P pair<int,int>

using ll = long long;

auto pick = [] (P a, P b) {
    if (a.first) b.second *= -1;
    return make_pair(a.first ^ b.first, a.second + b.second);
};

class SegTree {
public:
    int p = 2;
    vector<P> node;

    SegTree(vector<P> A) {
        while (A.size() > p) p *= 2;
        node = vector<P>(2*p-1, {0, 0});

        for (int i = 0; i < A.size(); i++) {
            node[p-1+i] = A[i];
        }

        for (int i = p - 2; i >= 0; i--) {
            node[i] = pick(node[i*2+1], node[i*2+2]);
        }
    }

    void update(int i, P v) {
        for (node[(i+=p)-1] = v; i >>= 1;) {
            node[i-1] = pick(node[i*2-1], node[i*2]);
        }
    }

    P getval(int start, int end, int k = 0, int l = 0, int r = -1) {
        if (r < 0) r = p;
        if (r <= start || end <= l) {return {0, 0};}
        if (start <= l && r <= end) {return node[k];}
        P L, R;
        L = getval(start, end, k*2+1, l, (l+r) / 2);
        R = getval(start, end, k*2+2, (l+r) / 2, r);
        return pick(L, R);
    }

    inline P operator [] (int i) {
        return node[i+p-1];
    }

    void debug() {
        rep (i, 2*p-1) {
            cout << node[i].first << "+" << node[i].second << " ";
        } cout << endl;
    }
};

signed main() {
    int k, n, q; cin >> k >> n >> q;
    vector<P> A(n);
    rep (i, n) {
        int a; cin >> a;
        if (a == 0) {
            A[i] = {1, 0};
        } else {
            A[i] = {0, a};
        }
    }
    vector<P> Q(q); rep (i, q) cin >> Q[i].first >> Q[i].second;

    SegTree seg(A);
    rep (i, q) {
        int l = Q[i].first - 1, r = Q[i].second - 1;
        seg.update(l, A[r]);
        seg.update(r, A[l]);
        swap(A[l], A[r]);
        P res = seg.getval(0, n);
        // rep (i, n) {
        //     cout << seg[i].second << " ";
        // } cout << endl;
        int ret = 1;

        if (res.second < 0) {
            ret -= res.second;
            ret %= k;
            if (ret == 0) ret += k;
        } else {
            ret -= res.second % k;
            ret += 2 * k; ret %= k;
            if (ret == 0) ret += k;
        }
        if (res.first == 1) ret *= -1;
        // seg.debug();
        // cout << res.second << " ";
        cout << ret << endl;
    }
}

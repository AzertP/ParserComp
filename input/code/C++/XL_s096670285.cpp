//#define _GLIBCXX_DEBUG
//#include "atcoder/all"
//using namespace atcoder;
#include <bits/stdc++.h>
#define int long long
#define ll long long
using ull = unsigned long long;
using namespace std;
#define dump(x)                             \
    if (dbg) {                              \
        cerr << #x << " = " << (x) << endl; \
    }
#define overload4(_1, _2, _3, _4, name, ...) name
#define FOR1(n) for (ll i = 0; i < (n); ++i)
#define FOR2(i, n) for (ll i = 0; i < (n); ++i)
#define FOR3(i, a, b) for (ll i = (a); i < (b); ++i)
#define FOR4(i, a, b, c) for (ll i = (a); i < (b); i += (c))
#define FOR(...) overload4(__VA_ARGS__, FOR4, FOR3, FOR2, FOR1)(__VA_ARGS__)
#define FORR(i, a, b) for (int i = (a); i <= (b); ++i)
#define bit(n, k) (((n) >> (k)) & 1) /*nのk bit目*/
namespace mydef {
const int INF = 1ll << 60;
const int MOD = 1e9 + 7;
template <class T>
bool chmin(T& a, const T& b) {
    if (a > b) {
        a = b;
        return 1;
    } else
        return 0;
}
template <class T>
bool chmax(T& a, const T& b) {
    if (a < b) {
        a = b;
        return 1;
    } else
        return 0;
}
void Yes(bool flag = true) {
    if (flag)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
}
void No(bool flag = true) {
    Yes(!flag);
}
void YES(bool flag = true) {
    if (flag)
        cout << "YES" << endl;
    else
        cout << "NO" << endl;
}
void NO(bool flag = true) {
    YES(!flag);
}
template <typename A, size_t N, typename T>
void Fill(A (&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}
bool dbg = true;
}  // namespace mydef
using namespace mydef;
#define pb push_back
//#define mp make_pair
#define eb emplace_back
#define lb lower_bound
#define ub upper_bound
#define all(v) (v).begin(), (v).end()
#define SZ(x) ((int)(x).size())
#define vi vector<int>
#define vvi vector<vector<int>>
#define vp vector<pair<int, int>>
#define vvp vector<vector<pair<int, int>>>
#define pi pair<int, int>
//#define P pair<int, int>
//#define V vector<int>
//#define S set<int>
#define asn ans

//LazySegmentTree(n, f, g, h, M1, OM0):= サイズ n の初期化。ここで f は2つの区間の要素をマージする二項演算, g は要素と作用素をマージする二項演算(第三引数は対応する区間の長さ), h は作用素同士をマージする二項演算, M1 はモノイドの単位元, OM0は作用素の単位元である。
//update(a, b, x) := 区間 [a,b) に作用素 x を適用する。
template <typename Monoid, typename OperatorMonoid = Monoid>
struct LazySegmentTree {
    using F = function<Monoid(Monoid, Monoid)>;
    using G = function<Monoid(Monoid, OperatorMonoid)>;
    using H = function<OperatorMonoid(OperatorMonoid, OperatorMonoid)>;

    int sz, height;
    vector<Monoid> data;
    vector<OperatorMonoid> lazy;
    const F f;
    const G g;
    const H h;
    const Monoid M1;
    const OperatorMonoid OM0;

    LazySegmentTree(int n, const F f, const G g, const H h,
        const Monoid& M1, const OperatorMonoid OM0)
        : f(f), g(g), h(h), M1(M1), OM0(OM0) {
        sz = 1;
        height = 0;
        while (sz < n)
            sz <<= 1, height++;
        data.assign(2 * sz, M1);
        lazy.assign(2 * sz, OM0);
    }

    void set(int k, const Monoid& x) {
        data[k + sz] = x;
    }

    void build() {
        for (int k = sz - 1; k > 0; k--) {
            data[k] = f(data[2 * k + 0], data[2 * k + 1]);
        }
    }

    inline void propagate(int k) {
        if (lazy[k] != OM0) {
            lazy[2 * k + 0] = h(lazy[2 * k + 0], lazy[k]);
            lazy[2 * k + 1] = h(lazy[2 * k + 1], lazy[k]);
            data[k] = reflect(k);
            lazy[k] = OM0;
        }
    }

    inline Monoid reflect(int k) {
        return lazy[k] == OM0 ? data[k] : g(data[k], lazy[k]);
    }

    inline void recalc(int k) {
        while (k >>= 1)
            data[k] = f(reflect(2 * k + 0), reflect(2 * k + 1));
    }

    inline void thrust(int k) {
        for (int i = height; i > 0; i--)
            propagate(k >> i);
    }

    void update(int a, int b, const OperatorMonoid& x) {
        thrust(a += sz);
        thrust(b += sz - 1);
        for (int l = a, r = b + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1)
                lazy[l] = h(lazy[l], x), ++l;
            if (r & 1)
                --r, lazy[r] = h(lazy[r], x);
        }
        recalc(a);
        recalc(b);
    }

    Monoid query(int a, int b) {
        thrust(a += sz);
        thrust(b += sz - 1);
        Monoid L = M1, R = M1;
        for (int l = a, r = b + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1)
                L = f(L, reflect(l++));
            if (r & 1)
                R = f(reflect(--r), R);
        }
        return f(L, R);
    }

    Monoid operator[](const int& k) {
        return query(k, k + 1);
    }

    template <typename C>
    int find_subtree(int a, const C& check, Monoid& M, bool type) {
        while (a < sz) {
            propagate(a);
            Monoid nxt = type ? f(reflect(2 * a + type), M) : f(M, reflect(2 * a + type));
            if (check(nxt))
                a = 2 * a + type;
            else
                M = nxt, a = 2 * a + 1 - type;
        }
        return a - sz;
    }

    template <typename C>
    int find_first(int a, const C& check) {
        Monoid L = M1;
        if (a <= 0) {
            if (check(f(L, reflect(1))))
                return find_subtree(1, check, L, false);
            return -1;
        }
        thrust(a + sz);
        int b = sz;
        for (a += sz, b += sz; a < b; a >>= 1, b >>= 1) {
            if (a & 1) {
                Monoid nxt = f(L, reflect(a));
                if (check(nxt))
                    return find_subtree(a, check, L, false);
                L = nxt;
                ++a;
            }
        }
        return -1;
    }


    template <typename C>
    int find_last(int b, const C& check) {
        Monoid R = M1;
        if (b >= sz) {
            if (check(f(reflect(1), R)))
                return find_subtree(1, check, R, true);
            return -1;
        }
        thrust(b + sz - 1);
        int a = sz;
        for (b += sz; a < b; a >>= 1, b >>= 1) {
            if (b & 1) {
                Monoid nxt = f(reflect(--b), R);
                if (check(nxt))
                    return find_subtree(b, check, R, true);
                R = nxt;
            }
        }
        return -1;
    }
};

//SegmentTree
//How to use
//SegmentTree(n, f, M1):= サイズ n の初期化。ここで f は2つの区間の要素をマージする二項演算, M1はモノイドの単位元である。
//set(k , x):= k 番目の要素に x を代入する。
//build():= セグメント木を構築する。
//query(a , b):= 区間 [a,b) に対して二項演算した結果を返す。
//update(k , x):= k 番目の要素を x に変更する。
//operator[k] : = k 番目の要素を返す。
template <typename Monoid>
struct SegmentTree {
    using F = function<Monoid(Monoid, Monoid)>;

    int sz;
    vector<Monoid> seg;

    const F f;
    const Monoid M1;

    SegmentTree(int n, const F f, const Monoid& M1) : f(f), M1(M1) {
        sz = 1;
        while (sz < n)
            sz <<= 1;
        seg.assign(2 * sz, M1);
    }

    void set(int k, const Monoid& x) {
        seg[k + sz] = x;
    }

    void build() {
        for (int k = sz - 1; k > 0; k--) {
            seg[k] = f(seg[2 * k + 0], seg[2 * k + 1]);
        }
    }

    void update(int k, const Monoid& x) {
        k += sz;
        seg[k] = x;
        while (k >>= 1) {
            seg[k] = f(seg[2 * k + 0], seg[2 * k + 1]);
        }
    }

    Monoid query(int a, int b) {
        Monoid L = M1, R = M1;
        for (a += sz, b += sz; a < b; a >>= 1, b >>= 1) {
            if (a & 1)
                L = f(L, seg[a++]);
            if (b & 1)
                R = f(seg[--b], R);
        }
        return f(L, R);
    }

    Monoid operator[](const int& k) const {
        return seg[k + sz];
    }
};
//example
/*
int main() {
  int N, Q;
  scanf("%d %d", &N, &Q);
  SegmentTree< int > seg(N, [](int a, int b) { return min(a, b); }, INT_MAX);
  while(Q--) {
    int T, X, Y;
    scanf("%d %d %d", &T, &X, &Y);
    if(T == 0) seg.update(X, Y);
    else printf("%d\n", seg.query(X, Y + 1));
  }
}
*/


int N, A[303030], K;

void solve() {
    /*
    LazySegmentTree<int,int> seg(303030,
            [](int a,int b){return max(a,b);},
            [](int a,int b))
            */
    SegmentTree<int> seg(303030, [](int a, int b) { return max(a, b); }, 0);
    seg.build();
    for (int i = 0; i < N; i++) {
        seg.update(A[i], 1ll + seg.query(max(0ll, A[i] - K), min(303000ll, A[i] + K + 1)));
    }
    cout << seg.query(0, 303000) << endl;
}

signed main() {
    cin.tie(nullptr);
    ios::sync_with_stdio(false);

    cin >> N >> K;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }

    solve();
    return 0;
}

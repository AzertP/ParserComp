// https://atcoder.jp/contests/abcXXX/tasks/abcXXX_x

#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using ll = long long;
using ull = unsigned long long;
using pll = pair<ll, ll>;

static const ll INF = 1001001001;

void debug_out() {};
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) {
    cerr << " " << to_string(H);
    debug_out(T...);
}
#ifdef LOCAL
#define debug(...) cerr << "[" << #__VA_ARGS__ << "]:", debug_out(__VA_ARGS__)
#else
#define debug(...) 1
#endif

struct C1 {
    vector<ll> v_{};
    explicit C1(vector<ll> v) : v_(move(v)){}
    explicit C1(ll N, vector<ll> r, vector<tuple<ll, ll, ll>> v) {}
    ll resolve() {
        return 0;
    }
};

int main() {
    ll A, B;
    cin >> A >> B;
    cout << std::max(0LL, A - 2 * B) << endl;
    return 0;
}

#include <bits/stdc++.h>
#define int int64_t
#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define REPP(i, n) for (int i = 1; (i) < (int)(n+1); ++ (i))
#define REPR(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define FOR(i, m, n) for(int i = m; i < n; i++)
#define FORR(i, m, n) for(int i=(m)-1; i>=(n); i--)
#define ALL(v) v.begin(), v.end()
#define SZ(a) int((a).size())
#define SORT(c) sort((c).begin(),(c).end())
#define SORTR(c) sort((c).begin(),(c).end(),greater<int>());
#define BIT(n) (1LL<<(n))
#define PCNT(x) __builtin_popcountll(x)
#define GCD(x,y) __gcd(x,y)
#define LCM(x,y) x/__gcd(x,y)*y
#define PB push_back
#define MP make_pair
#define endl "\n"
#define pa cout << ans << endl
#define p(x) cout << x << endl
#define pall(x) for(auto iter : x) p(iter)
#define yes cout << "Yes" << endl
#define no cout << "No" << endl
#define yn(x) cout << (x ? "Yes" : "No") << endl
using namespace std;
using Graph = vector<vector<int> >;
using WeightedGraph = vector<vector<pair<int, int> > >;
typedef long double LD;
typedef vector<int> VI;
typedef vector<vector<int> > VVI;
typedef vector<string> VS;
typedef pair<int, int> PII;
typedef vector<PII> VPII;
const int INF = 2000000000;
const long double PI = acosl(-1);
const int MOD = 1000000007;
void print() {cout << endl;}
template <class Head, class... Tail> void print(Head&& head, Tail&&... tail) {cout << head; if (sizeof...(tail) != 0) cout << " ";print(forward<Tail>(tail)...);}
template <class T> void print(vector<T> &vec) {for (auto& a : vec) {cout << a; if (&a != &vec.back()) cout << " ";} cout << endl;}
template <class T> void print(vector<vector<T>> &df) { for (auto& vec : df) { print(vec);}}


signed main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    cout << fixed << setprecision(10);
    int ans = 0;
    
    int N;
    cin >> N;

    if (N%1000 == 0) ans = 0;
    else ans = 1000-(N%1000);
    cout << ans << endl;
    
    
    
    return 0;
}

/*
cd ~/Desktop/Programming/AtCoder/contest/abc173/abc173_a && sh shellscripts/test_cpp.sh
cd ~/Desktop/Programming/AtCoder/contest/abc173/abc173_a && sh shellscripts/etest_cpp.sh
cd ~/Desktop/Programming/AtCoder/contest/abc173/abc173_a && sh shellscripts/submit_cpp.sh
*/

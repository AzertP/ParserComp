#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pll = pair<ll, ll>;
using vl = vector<ll>;

/* short */
#define pb push_back
#define mp make_pair
#define Fi first
#define Se second
#define ALL(v) begin(v), end(v)
#define RALL(v) rbegin(v), rend(v)

/* REPmacro */
#define FOR(i, a, b) for(int i = (a); i < (b); i++)
#define FORR(i, a, b) for(int i = (a); i >= (b); i--)
#define REP(i, n) for(int i = 0; i < (n); i++)
#define FOREACH(x, a) for(auto x : a)

/* exchange */
#define CHMIN(a, b) (a) = min((ll)(a), (ll)(b))
#define CHMAX(a, b) (a) = max((ll)(a), (ll)(b))

/* function */
#define IN(x) cin >> x
#define DEBUG(x) cerr << (x) << " "
#define LN() cerr << "\n"
#define PRINT(x) cout << (x) << endl
#define BR cout << endl

/* const */
const int ARRAY = 100005;
const int INF = 1001001001; // 10^9
const ll LINF = 1001001001001001001; // 10^18
const int MOD = 1e9 + 7;

ll N = 0;
ll ret = 0;
string s;

struct Node {
  vl adj;
};

void dfs(ll init, vl& dist, vector<bool>& checked, vector<Node>& nodes) {
  stack<ll> s;
  s.push(init);
  dist[init] = 0;
  checked[init] = true;
  while(!s.empty()) {
    ll t = s.top();
    s.pop();
    FOREACH(k, nodes[t].adj) {
      if (!checked[k]) {
        dist[k] = dist[t] + 1;
        checked[k] = true;
        s.push(k);
      }
    }
  }
}

// bool compare(pair<ll, ll> a, pair<ll, ll> b) {
//   if (a.first != b.first) {
//     return a.first > b.first;
//   } else {
//     return a.second < b.second;
//   }
// }

int main(void){
  ll u, v;
  IN(N);
  IN(u);
  IN(v);
  
  vector<Node> nodes(N+1);
  REP(i, N) {
    ll a, b;
    IN(a);
    IN(b);
    nodes[a].adj.pb(b);
    nodes[b].adj.pb(a);
  }
  vector<bool> checkedT(N+1, false);
  vector<bool> checkedA(N+1, false);
  vl distT(N+1);
  vl distA(N+1);
  dfs(u, distT, checkedT, nodes);
  dfs(v, distA, checkedA, nodes);

  // FOR(i, 1, N+1) DEBUG(distT[i]);
  // LN();
  // FOR(i, 1, N+1) DEBUG(distA[i]);
  // LN();

  // vector<pair<ll, ll>> A;
  vl A;
  FOR(i, 1, N+1) {
    if (distT[i] <= distA[i]) {
      A.pb(distA[i] - 1);
    }
  }
  sort(ALL(A));
  reverse(ALL(A));
  PRINT(A[0]);
}

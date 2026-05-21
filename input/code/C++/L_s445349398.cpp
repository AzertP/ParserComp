#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>

#define FOR(i, a, b) for(int i=(int)a; i < (int)b; ++i)
#define REP(i, n) FOR(i,0,n)
#define RFOR(i, a, b) for(int i=(int)b-1; i >= (int)a; --i)
#define RREP(i, n) RFOR(i,0,n)
#define IN(a,x,b) (a<=x && x < b)
template<class T> inline void chmax(T& a, const T& b){if(a<b) a = b;}
template<class T> inline void chmin(T& a, const T& b){if(a>b) a = b;}

using namespace std;
using ll = long long;
template<class T> using V = std::vector<T>;
template<class T> using VV = V<V<T>>;

constexpr int INFINT = (1 << 30) - 1;
int cost[22][20004];
int g[22][102][102];
int G[102][102];

signed main(){
  int n, m, C, s, g_;
  while(std::cin >> n >> m >> C >> s >> g_, n || m || C || s || g_) {
    --s; --g_;
    V<int> x(m), y(m), d(m), c(m);
    REP(i, m) std::cin >> x[i] >> y[i] >> d[i] >> c[i], --x[i], --y[i], --c[i];
    V<int> p(C);
    REP(i, C) std::cin >> p[i];
    VV<int> q(C), r(C);
    REP(i, C) {
      q[i].resize(p[i] - 1);
      r[i].resize(p[i]);
      REP(j, p[i] - 1) std::cin >> q[i][j];
      q[i].emplace_back(INFINT);
      REP(j, p[i]) std::cin >> r[i][j];
    }

    { // fill cost[][]
        REP(j, C) {
          cost[j][0] = 0;
          int pos = 0;
          FOR(d, 1, 20004) {
            cost[j][d] = cost[j][d-1] + r[j][pos];
            if(d == q[j][pos]) ++pos;
          }
        }
    }

    REP(k, C) {
      REP(i, n) REP(j, n) g[k][i][j] = INFINT;
      REP(i, n) g[k][i][i] = 0;
      REP(i, m) {
        if(c[i] != k) continue;
        chmin(g[k][x[i]][y[i]], d[i]);
        chmin(g[k][y[i]][x[i]], d[i]);
      }
    }

    REP(p, C) REP(k, n) REP(i, n) REP(j, n) {
      chmin(g[p][i][j], g[p][i][k] + g[p][k][j]);
    }

    REP(p, C) REP(i, n) REP(j, n) {
      if(g[p][i][j] >= INFINT) continue;
      g[p][i][j] = cost[p][g[p][i][j]];
    }

    REP(i, n) REP(j, n) G[i][j] = INFINT;
    REP(i, n) G[i][i] = 0;

    REP(p, C) REP(i, n) REP(j, n) chmin(G[i][j], g[p][i][j]);

    REP(k, n) REP(i, n) REP(j, n) chmin(G[i][j], G[i][k] + G[k][j]);

    int ans = G[s][g_];
    if(ans >= INFINT) ans = -1;
    std::cout << ans << std::endl;
  }
  return 0;
}


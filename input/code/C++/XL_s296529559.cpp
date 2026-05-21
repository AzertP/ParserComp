#pragma region Macros
#include <bits/stdc++.h>
using namespace std;
//#include <boost/multiprecision/cpp_int.hpp>
//using multiInt = boost::multiprecision::cpp_int;

using ll = long long int;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using pld = pair<ld, ld>;
template <typename Q_type>
using smaller_queue = priority_queue<Q_type, vector<Q_type>, greater<Q_type>>;

constexpr int MOD_TYPE = 1;
constexpr ll MOD = (MOD_TYPE == 1 ? (ll)(1e9 + 7) : 998244353);
constexpr int INF = (int)1e9;
constexpr ll LINF = (ll)4e18;
constexpr ld PI = acos(-1.0);
constexpr ld EPS = 1e-11;
constexpr int Dx[] = {0, 0, -1, 1, -1, 1, -1, 1, 0};
constexpr int Dy[] = {1, -1, 0, 0, -1, -1, 1, 1, 0};

#define REP(i, m, n) for (ll i = m; i < (ll)(n); ++i)
#define rep(i, n) REP(i, 0, n)
#define MP make_pair
#define MT make_tuple
#define YES(n) cout << ((n) ? "YES" : "NO") << "\n"
#define Yes(n) cout << ((n) ? "Yes" : "No") << "\n"
#define possible(n) cout << ((n) ? "possible" : "impossible") << "\n"
#define Possible(n) cout << ((n) ? "Possible" : "Impossible") << "\n"
#define Yay(n) cout << ((n) ? "Yay!" : ":(") << "\n"
#define all(v) v.begin(), v.end()
#define NP(v) next_permutation(all(v))
#define dbg(x) cerr << #x << ":" << x << "\n";

#pragma endregion

struct point_t
{
  ld x, y;
  int exception = 0;

  inline void display()
  {
    if (exception == 0)
      cerr << "(" << x << ", " << y << ")\n";
    else
      cerr << "exception:" << exception << "\n";
  }

  constexpr point_t operator+(const point_t &p) const noexcept
  {
    return point_t{this->x + p.x, this->y + p.y};
  }
  constexpr point_t operator-(const point_t &p) const noexcept
  {
    return point_t{this->x - p.x, this->y - p.y};
  }
  constexpr point_t operator*(const ld &a) const noexcept
  {
    return point_t{this->x * a, this->y * a};
  }
  constexpr point_t operator/(const ld &a) const noexcept
  {
    return point_t{this->x / a, this->y / a};
  }
};

struct line_t
{
  ld a, b, c;
  int exception = 0;

  inline void display()
  {
    if (exception == 0)
      cerr << a << 'x'
           << (b < 0 ? "" : "+") << b << "y"
           << (c < 0 ? "" : "+") << c << "=0\n";
    else
      cerr << "exception:" << exception << "\n";
  }
};

struct circle_t
{
  point_t ce;
  ld r;
  int exception = 0;

  inline void display()
  {
    if (exception == 0)
    {
      cerr << "center:";
      ce.display();
      cerr << "r:" << r << "\n";
    }
    else
      cerr << "exception:" << exception << "\n";
  }
};

struct rect_t
{
  point_t l, r; //左下，右上
  int exception = 0;

  inline void display()
  {
    if (exception == 0)
    {
      cerr << "l:";
      l.display();
      cerr << "r:";
      r.display();
    }
    else
      cerr << "exception:" << exception << "\n";
  }
};

using pp_t = pair<point_t, point_t>;

inline bool Same(point_t &p1, point_t &p2)
{
  return (abs(p1.x - p2.x) < EPS && abs(p1.y - p2.y) < EPS);
}

bool Same(line_t &l1, line_t &l2)
{
  bool b1 = abs(l1.a * l2.b - l1.b * l2.a) < EPS;
  bool b2 = abs(l1.b * l2.c - l1.c * l2.b) < EPS;
  bool b3 = abs(l1.c * l2.a - l1.a * l2.c) < EPS;
  return (b1 && b2 && b3);
}

inline bool Same(circle_t &c1, circle_t &c2)
{
  return (Same(c1.ce, c2.ce) && abs(c1.r - c2.r) < EPS);
}

inline bool PointOn(point_t &p, line_t &l)
{
  return abs(l.a * p.x + l.b * p.y + l.c) < EPS;
}

inline bool PointOn(point_t &p, circle_t &c)
{
  return abs((p.x - c.ce.x) * (p.x - c.ce.x) + (p.y - c.ce.y) * (p.y - c.ce.y) - c.r * c.r) < EPS;
}

inline bool Parallel(line_t &l1, line_t &l2)
{
  return abs(l1.a * l2.b - l1.b * l2.a) < EPS;
}

inline ld Dist(point_t &p1, point_t &p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

inline ld Dist(point_t &p, line_t &l)
{
  return abs(l.a * p.x + l.b * p.y + l.c) / sqrt(l.a * l.a + l.b * l.b);
}

//M,Nに内分
inline point_t InternallyDivide(point_t &p1, point_t &p2, ld M, ld N)
{
  return (p1 * N + p2 * M) / (M + N);
}

//M,Nに外分
inline point_t ExternallyDivide(point_t &p1, point_t &p2, ld M, ld N)
{
  return (p1 * (-N) + p2 * M) / (M - N);
}

//中点
inline point_t MidP(point_t &p1, point_t &p2)
{
  return (p1 + p2) / 2.0;
}

//点の回転
inline point_t Rotate(point_t &p, ld theta)
{
  return point_t{p.x * cos(theta) - p.y * sin(theta), p.x * sin(theta) + p.y * cos(theta)};
}

//2点を通る直線の方程式
//p1 = p2の場合はexception = 1
line_t Line2p(point_t &p1, point_t &p2)
{
  if (Same(p1, p2))
    return line_t{0, 0, 0, 1};
  line_t res;
  res.a = (p2.y - p1.y);
  res.b = (p1.x - p2.x);
  res.c = (p2.x - p1.x) * p1.y - (p2.y - p1.y) * p1.x;
  return res;
}

//2直線の交点
//2直線が一致する場合はexception = 1
//交わらない場合はexception = 2
point_t Intersection(line_t &l1, line_t &l2)
{
  if (Parallel(l1, l2))
    return Same(l1, l2) ? point_t{0, 0, 1} : point_t{0, 0, 2};
  point_t res;
  res.x = (l1.b * l2.c - l1.c * l2.b) / (l1.a * l2.b - l1.b * l2.a);
  res.y = (l1.a * l2.c - l1.c * l2.a) / (l1.b * l2.a - l1.a * l2.b);
  return res;
}

//pを通りlに垂直な直線
line_t VerticalLine(point_t &p, line_t &l)
{
  line_t res;
  res.a = l.b;
  res.b = -l.a;
  res.c = l.a * p.y - l.b * p.x;
  return res;
}

//p1,p2を端点とする線分の垂直二等分線
//2点が一致する場合はexception = 1
line_t VerticalBisector(point_t &p1, point_t &p2)
{
  if (Same(p1, p2))
    return line_t{0, 0, 0, 1};
  point_t m = MidP(p1, p2);
  line_t l = Line2p(p1, p2);
  return VerticalLine(m, l);
}

//lに関してpと対称な点
point_t Reflect(point_t &p, line_t &l)
{
  ld s = l.a * p.x + l.b * p.y + 2 * l.c;
  ld t = l.b * p.x - l.a * p.y;
  ld u = l.a * l.a + l.b * l.b;
  point_t res;
  res.x = (l.b * t - l.a * s) / u;
  res.y = (-l.b * s - l.a * t) / u;
  return res;
}

//pがcの内部なら1
//周上なら0
//外部なら-1
int InCircle(point_t &p, circle_t &c)
{
  if (abs((p.x - c.ce.x) * (p.x - c.ce.x) + (p.y - c.ce.y) * (p.y - c.ce.y) - c.r * c.r) < EPS)
    return 0;
  else if ((p.x - c.ce.x) * (p.x - c.ce.x) + (p.y - c.ce.y) * (p.y - c.ce.y) - c.r * c.r < 0.0)
    return 1;
  else
    return -1;
}

//3点を通る円
//いずれかの2点が一致するならexception = 1
//3点が同一直線上にあるならexception = 2
circle_t Circle3p(point_t &p1, point_t &p2, point_t &p3)
{
  if (Same(p1, p2) || Same(p2, p3) || Same(p3, p1))
    return circle_t{point_t{0, 0}, 0, 1};
  line_t l1 = VerticalBisector(p1, p2);
  line_t l2 = VerticalBisector(p1, p3);
  point_t center = Intersection(l1, l2);
  if (center.exception == 2)
    return circle_t{point_t{0, 0}, 0, 2};
  circle_t res;
  res.ce = center;
  res.r = Dist(center, p1);
  return res;
}

//2円の共有点の個数
//2円が重なるときは3
int CommonP(circle_t &c1, circle_t &c2)
{
  if (Same(c1, c2))
    return 3;
  ld d = Dist(c1.ce, c2.ce);
  if (abs(d - c1.r - c2.r) < EPS || abs(d - c1.r + c2.r) < EPS)
    return 1;
  else if (d > c1.r + c2.r || d < abs(c1.r - c2.r))
    return 0;
  else
    return 2;
}

//2円の2個の共有点を通る直線
//2円が接する場合は接線
//共有点がない場合はexception = 1
line_t CCLine(circle_t &c1, circle_t &c2)
{
  line_t l;
  l.a = 2.0 * (c2.ce.x - c1.ce.x);
  l.b = 2.0 * (c2.ce.y - c1.ce.y);
  l.c = c1.ce.x * c1.ce.x - c2.ce.x * c2.ce.x;
  l.c += c1.ce.y * c1.ce.y - c2.ce.y * c2.ce.y;
  l.c += -c1.r * c1.r + c2.r * c2.r;
  return l;
}

//2点からの距離の比がM:Nの点の軌跡（アポロニウスの円）
//M=Nの場合はexception = 1
//2点が重なる場合はexception = 2
circle_t Apollonius(point_t &p1, point_t &p2, ld M, ld N)
{
  if (abs(M - N) < EPS)
    return circle_t{point_t{0, 0}, 0, 1};
  circle_t res;
  res.ce = ExternallyDivide(p1, p2, M * M, N * N);
  res.r = sqrt(Dist(res.ce, p1) * Dist(res.ce, p2));
  return res;
}

//円と直線の交点
//交点がない場合はexception = 1
pp_t Intersection(circle_t &c, line_t &l)
{
  if (Dist(c.ce, l) > c.r + EPS)
    return make_pair(point_t{0, 0, 1}, point_t{0, 0, 1});
  point_t res1, res2;
  ld d = l.a * c.ce.x + l.b * c.ce.y + l.c;
  ld sq = sqrt((l.a * l.a + l.b * l.b) * c.r * c.r - d * d);
  res1.x = c.ce.x + (-l.a * d + l.b * sq) / (l.a * l.a + l.b * l.b);
  res2.x = c.ce.x + (-l.a * d - l.b * sq) / (l.a * l.a + l.b * l.b);
  res1.y = c.ce.y + (-l.b * d - l.a * sq) / (l.a * l.a + l.b * l.b);
  res2.y = c.ce.y + (-l.b * d + l.a * sq) / (l.a * l.a + l.b * l.b);
  return make_pair(res1, res2);
}

//交点がない場合はexception = 1
//2円が一致する場合はexception = 2
pp_t Intersection(circle_t &c1, circle_t &c2)
{
  if (CommonP(c1, c2) == 0)
    return make_pair(point_t{0, 0, 1}, point_t{0, 0, 1});
  else if (Same(c1, c2))
    return make_pair(point_t{0, 0, 2}, point_t{0, 0, 2});
  point_t shift = c1.ce;
  c1.ce = c1.ce - shift, c2.ce = c2.ce - shift;
  line_t l = CCLine(c1, c2);
  pp_t res = Intersection(c1, l);
  res.first = res.first + shift;
  res.second = res.second + shift;
  return res;
}

//長方形の共通部分
//存在しない場合はexception = 1
rect_t Intersection(rect_t &r1, rect_t &r2)
{
  rect_t res;

  pair<ld, int> sx[4] = {
      make_pair(r1.l.x, 0),
      make_pair(r1.r.x, 1),
      make_pair(r2.l.x, 2),
      make_pair(r2.r.x, 3),
  };
  sort(sx, sx + 4);
  bool b1 = (sx[1].second == 1 && sx[2].second == 2 && sx[1].first < sx[2].first);
  bool b2 = (sx[1].second == 3 && sx[2].second == 0 && sx[1].first < sx[2].first);
  if (b1 || b2)
    return rect_t{{0, 0}, {0, 0}, 1};
  res.l.x = sx[1].first;
  res.r.x = sx[2].first;

  pair<ld, int> sy[4] = {
      make_pair(r1.l.y, 0),
      make_pair(r1.r.y, 1),
      make_pair(r2.l.y, 2),
      make_pair(r2.r.y, 3),
  };
  sort(sy, sy + 4);
  b1 = (sy[1].second == 1 && sy[2].second == 2 && sy[1].first < sy[2].first);
  b2 = (sy[1].second == 3 && sy[2].second == 0 && sy[1].first < sy[2].first);
  if (b1 || b2)
    return rect_t{{0, 0}, {0, 0}, 1};
  res.l.y = sy[1].first;
  res.r.y = sy[2].first;

  return res;
}

int main()
{
  cin.tie(0);
  ios::sync_with_stdio(false);
  cout << setprecision(30) << setiosflags(ios::fixed);

  int n;
  cin >> n;
  point_t p[50];
  rep(i, n)
  {
    cin >> p[i].x >> p[i].y;
  }

  ld Min = INF;
  rep(i, n) REP(j, i + 1, n)
  {
    point_t mid = MidP(p[i], p[j]);
    ld Max = 0;
    rep(t, n) Max = max(Max, Dist(mid, p[t]));
    Min = min(Min, Max);
  }

  rep(i, n) REP(j, i + 1, n) REP(k, j + 1, n)
  {
    circle_t c = Circle3p(p[i], p[j], p[k]);
    if (c.exception != 0)
      continue;
    ld Max = 0;
    rep(t, n) Max = max(Max, Dist(c.ce, p[t]));
    Min = min(Min, Max);
  }
  cout << Min << endl;
  return 0;
}

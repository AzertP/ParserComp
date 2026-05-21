//ProblemF.cpp

#include <iostream>

static std::istream & ip = std::cin;
static std::ostream & op = std::cout;

#if OJ_MYPC
#include <ojio.h>
#endif

#ifndef OPENOJIO
#define OPENOJIO
#endif

#if 1 || DEFINE
/***************************************************************/
typedef unsigned long long u64;
typedef long long s64;

typedef unsigned uint;

#define ABS(x) ((x) > 0 ? (x) : -(x))

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define MIN3(x, y, z) MIN(x, MIN(y, z))
#define MAX3(x, y, z) MAX(x, MAX(y, z))

#define FillZero(arr) memset(arr, 0, sizeof(arr));

/***************************************************************/
#endif //1 || DEFINE

#include <string>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <queue>
#include <stack>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cstdio>

#include <functional> 
#include <assert.h>

//001
//op << setfill('0') << setw(3) << setiosflags(ios::right) << 1;

//op << fixed << setprecision(20);

using namespace std;

//ProblemF.cpp

#define MAXN 200010
#define MOD 924844033LL

#define N (1 << 19)
#define W 521458

s64 fpow(s64 a, s64 x, s64 m = MOD) {
	if (x == 0) return 1;
	a = a % m; if (a < 0) a += m;

	s64 mul = a;
	s64 rst = 1;
	while (x)
	{
		if (x & 1) rst = rst * mul % m;

		x >>= 1;
		mul = mul * mul % m;
	}
	return rst;
}

class FFT {
	const int _n;
	const s64 _w;
	const s64 _w_inv;

public:
	FFT(int n, s64 w) : _n(n), _w(w), _w_inv(fpow(w, n - 1)) {
		assert(_w * _w_inv % MOD == 1);
		for(--n; n; n >>= 1) assert(n & 1);
	}

	void calc(vector<s64>& rst, const vector<s64>& p, bool inv = false) {
		assert(p.size() <= _n);
		rst = p;

		calc(rst, inv);
	}

	void calc(vector<s64>& p, bool inv = false) {
		assert(p.size() <= _n);
		p.resize(_n);

		const s64 w = inv ? _w_inv : _w;
		s64 wm = w;
		for (int m = _n; m >= 2; m >>= 1) {

			int d = m >> 1;
			s64 wj = 1;
			for (int i = 0; i < d; i++) {
				for (int j = i; j < _n; j += m) {
					int k = j + d;
					s64 tmp = (MOD + p[j] - p[k]) % MOD;

					(p[j] += p[k]) %= MOD;
					p[k] = wj * tmp % MOD;

				}
				(wj *= wm) %= MOD;
			}
			(wm *= wm) %= MOD;
		}
		int i = 0;
		for (int j = 1; j < _n - 1; j++) {
			for (int k = _n >> 1; k > (i ^= k); k >>= 1);
			if (j < i) swap(p[i], p[j]);
		}
	}
};

//gcdex = ax + by
template <typename Int>
Int gcdex(Int a, Int b, Int& x, Int& y)
{
	bool swapped = false;

	if (a < b) { swap(a, b); swapped = true; }
	Int rst = 0;

	if (a % b == 0) { x = 0; y = 1; rst = b; }
	else
	{
		Int tx, ty;
		rst = gcdex(b, a % b, tx, ty);

		x = ty;
		y = tx - ty * (a / b);
	}

	if (swapped) swap(x, y);
	return rst;
}

int main(int argc, char* argv[])
{
	OPENOJIO;

	static s64 fac[MAXN + 1];
	fac[0] = fac[1] = 1;
	for (int i = 2; i <= MAXN; ++i) fac[i] = fac[i - 1] * i % MOD;

	static s64 fac_inv[MAXN + 1];
	fac_inv[0] = fac_inv[1] = 1;
	for (int i = 2; i <= MAXN; ++i) {
		s64 tmp;
		gcdex(MOD, fac[i], tmp, fac_inv[i]);
		fac_inv[i] %= MOD;
		if (fac_inv[i] < 0) fac_inv[i] += MOD;
		assert(fac[i] * fac_inv[i] % MOD == 1);
	}

	int n;
	vector<vector<int> > nears;

	ip >> n;

	nears.resize(n);
	for (int i = 0; i < n - 1; ++i) {
		int a, b;
		ip >> a >> b;
		--a; --b;

		nears[a].push_back(b);
		nears[b].push_back(a);
	}

	vector<int> count_sons;
	count_sons.resize(n, 0);

	function<int (int, int)> get_count_sons = [&] (int cur, int father) -> int {
		count_sons[cur] = 1;
		for (auto son : nears[cur]) {
			if (son == father) continue;
			count_sons[cur] += get_count_sons(son, cur);
		}
		return count_sons[cur];
	};

	get_count_sons(0, -1);

	vector<int> b;
	b.resize(n, 0);

	for (int i = 1; i < n; ++i) {
		b[count_sons[i]]++;
		b[n - count_sons[i]]++;
	}

	vector<s64> c(N, 0);
	vector<s64> d(N, 0);

	for (int i = 0; i < n; ++i)
		c[i] = b[i] * fac[i] % MOD;

	d[0] = 1;
	for (int i = 1; i < n; ++i)
		d[N - i] = fac_inv[i] % MOD;

	FFT fft(N, W);
	fft.calc(c);
	fft.calc(d);

	for (int i = 0; i < c.size(); ++i) (c[i] *= d[i]) %= MOD;
	fft.calc(c, true);
	s64 N_inv = fpow(N, MOD - 2);
	assert(N * N_inv % MOD == 1);
	for (int i = 0; i < c.size(); ++i) (c[i] *= N_inv) %= MOD;

	vector<s64> &rst = d;
	for (int i = 1; i <= n; ++i)
	{
		rst[i] = ((n * fac[n] % MOD * fac_inv[i] % MOD * fac_inv[n - i] - fac_inv[i] * c[i]) % MOD + MOD) % MOD;
	}

	for (int i = 1; i <= n; ++i)
	{
		op << rst[i] << endl;
	}

	return 0;
}
/***************************************************************/

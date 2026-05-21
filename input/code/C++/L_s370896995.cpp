#include <iostream>
#include <set>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstring>
#include <climits>
#include <sstream>
#include <iomanip>
#include <map>
#include <stack>
#include <numeric>

using namespace std;

/*-----------------------------------------------------------------------------
　定義
 -------------------------------------------------------------------------------*/
#define ALL(x)					(x).begin(),(x).end()
#define REP(i, n)				for (int (i) = 0 ; (i) < (ll)(n) ; ++(i))
#define REPN(i, m, n)			for (int (i) = m ; (i) < (ll)(n) ; ++(i))
#define INF						(int)2e9
#define MOD						(1000 * 1000 * 1000 + 7)
#define Ceil(x, n)				(((((x))+((n)-1))/n))		/* Nの倍数に切り上げ割り算 */
#define CeilN(x, n)				(((((x))+((n)-1))/n)*n)		/* Nの倍数に切り上げ */
#define FloorN(x, n)			((x)-(x)%(n))				/* Nの倍数に切り下げ */
#define IsOdd(x)				(((x)&0x01UL) == 0x01UL)			
#define IsEven(x)				(!IsOdd((x)))						
#define M_PI					3.14159265358979323846
typedef long long				ll;
typedef pair<ll, ll>			P;

/*-----------------------------------------------------------------------------
　処理
 -------------------------------------------------------------------------------*/
int main()
{
	ll N, C;
	cin >> N >> C;
	vector<ll> x(N), v(N);
	REP(i, N) cin >> x[i] >> v[i];

	vector<ll> rvMax(N, 0);
	ll nowPos = 0;
	ll calTotal = 0;
	ll ans1 = 0;
	REP(i, N) {
		ll calOne = v[i] - (x[i] - nowPos);
		calTotal += calOne;
		ans1 = max(calTotal, ans1);
		rvMax[i] = max(rvMax[i], ans1);
		nowPos = x[i];
	}

	vector<ll> lvMax(N, 0);
	nowPos = C;
	calTotal = 0;
	ll ans2 = 0;
	for (int i = N - 1; i >= 0; i--) {
		ll calOne = v[i] - (nowPos - x[i]);
		calTotal += calOne;
		ans2 = max(calTotal, ans2);
		lvMax[i] = max(lvMax[i], ans2);
		nowPos = x[i];
	}

	ll ans3 = rvMax[N - 1];
	for (int i = 0; i < N - 1; i++) {
		ll calOne = rvMax[i] + lvMax[i + 1] - x[i];
		ans3 = max(ans3, calOne);
	}
	
	ll ans4 = lvMax[0];
	for (int i = 0; i < N - 1; i++) {
		ll calOne = rvMax[i] + lvMax[i + 1] - (C - x[i + 1]);
		ans4 = max(ans4, calOne);
	}

 	cout << max(ans3, ans4) << endl;
	return 0;
}

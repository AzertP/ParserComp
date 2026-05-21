#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <stdio.h>
#include <string>
#include <vector>
#include <set>

#define rep(i,n) for (int i = 0; i < (n); ++i)
using namespace std;
using ll = long long;
using P = pair<int, int>;
const double PI = 3.1415926535897932;

template<class T> inline bool chmin(T& a, T b) {
	if (a > b) {
		a = b;
		return true;
	}
	return false;
}

template<class T> inline bool chmax(T& a, T b) {
	if (a < b) {
		a = b;
		return true;
	}
	return false;
}

int N;
const long long INF = 1LL << 60;
long long h[100010];
long long dp[100010];

int main()
{
	int N; cin >> N;
	for (int i = 0; i < N; ++i) cin >> h[i];

	// 初期化 (最小化問題なので INF に初期化)
	for (int i = 0; i < 100010; ++i) dp[i] = INF;

	// 初期条件
	dp[0] = 0;

	dp[0] = 0;
	for (int i = 1; i < N; ++i) {
		chmin(dp[i], dp[i - 1] + abs(h[i] - h[i - 1]));
		if (i > 1) chmin(dp[i], dp[i - 2] + abs(h[i] - h[i - 2]));
	}

	cout << dp[N-1] << endl;
	return 0;
}
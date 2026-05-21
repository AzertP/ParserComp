#include<stdio.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iterator>
#include <list>
#include <map>     
#include <numeric>
#include <utility>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <stack>
#include <string>
#include <vector>
using namespace std;
#define ll long long
#define ld long double
#define rep(a,t) for(int a=0;a<t;++a) 
#define forever while(true)
#define Sort(a) sort(a.begin(),a.end())
#define Reverse(a) reverse(a.begin(),a.end())
#define pb push_back
#define int_maxvalue numeric_limits<int>::max()
#define print_double(val,a) cout << fixed << setprecision(a) << val << endl;
ll mod = 1e9 + 7;
vector<int>era(int n) {
	bool* dp = new bool[n];
	for (int i = 0; i < n; i++) dp[i] = false;
	dp[0] = dp[1] = true;
	vector<int>prime;
	for (int i = 2; i < n; i++) {
		if (dp[i]) continue;
		prime.push_back(i);
		for (int j = i * 2; j < n; j += i) {
			dp[j] = true;
		}
	}
	return prime;
}
int main()
{
	cin.tie(0);
	ios::sync_with_stdio(false);
	int n;
	cin >> n;
	vector<int>prime = era(55555);
	int ans=0;
	for (int i = 0; i < prime.size(); i++) {
		if (ans == n) { break; }
		if (prime[i] % 5 == 1) {
			if (ans != 0) { cout << " "; }
			cout << prime[i];
			ans++;
		}
	}
	return 0;
}
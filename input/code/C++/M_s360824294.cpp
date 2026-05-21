#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

#define ll long long
#define INF (1 << 30)
#define INFLL (1LL << 60)

#define FOR(i,a,b) for(ll i = (a);i<(b);i++)
#define REP(i,a) FOR(i,0,(a))
#define MP make_pair

int main() {
	int sum = 0;
	int n,t[101];
	cin >> n;
	REP(i, n){
		cin >> t[i];
		sum += t[i];
	}
	int m,p,x;
	cin >> m;
	REP(i, m){
		cin >> p >> x;
		p--;
		cout << sum - t[p] + x << endl;
	}

	return 0;
}
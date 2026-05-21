#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cassert>
#include <string>
#include <algorithm>
#include <vector>
#include <queue>
#include <stack>
#include <functional>
#include <iostream>
#include <map>
#include <set>
using namespace std;
typedef pair<int,int> P;
typedef pair<int,P> P1;
typedef pair<P,P> P2;
typedef long long ll;
#define pb push_back
#define mp make_pair
#define eps 1e-7
#define INF 1000000000
#define fi first
#define sc second
#define rep(i,x) for(int i=0;i<x;i++)

int main()
{
	while(1)
	{
		int n;
		ll res = 0;
		ll sum = 0;
		vector<ll>con;
		
		cin >> n;
		if(n == 0) break;
		for(int i=0;i<n;i++)
		{
			ll a; cin >> a;
			sum+=a;
		}
		for(int i=0;i<n-1;i++)
		{
			ll b; cin >> b;
			con.pb(b); sum+=b;
		}
		sort(con.begin(),con.end());
		res = sum;
		
		for(int i=0;i<con.size();i++)
		{
			sum -= con[i];
			res = max(res,sum*1LL*(i+2));
		}
		
		cout << res << endl;
	}
}
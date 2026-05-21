#include<iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <set>

#define rep(i,n)  for (int i = 0; i < (int)(n); i++)
#define INF 9999999999
#define PI 3.14159265359
using namespace std;

int main()
{
	string s,t;
	cin >>s>>t;
	int i,ans=0;
	rep(i,s.size()){
		if (s[i]!=t[i]) ans++;
	}
	cout << ans << endl;
	return 0;
}

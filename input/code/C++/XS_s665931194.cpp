#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define pb pushback
#define fr(i,n) for(int i=0;i<n;i++)
#define ifr(i,n) for(int i=n-1;i>=0;i--)


int main() {
    int a,b,c;
	cin >> a>> b>> c;
    int ans=3;
    if(a==b&&b==c)ans=1;
    else if(a==b||b==c||c==a)ans=2;
      cout << ans << endl;
}
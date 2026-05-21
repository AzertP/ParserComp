#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<ll,ll> pi;
typedef vector <ll> vi;
typedef vector <pi> vpi;
#define f first
#define s second
#define FOR(i,s,e) for(ll i=s;i<=ll(e);++i)
#define DEC(i,s,e) for(ll i=s;i>=ll(e);--i)
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define lbd(x, y) lower_bound(all(x), y)
#define ubd(x, y) upper_bound(all(x), y)
#define aFOR(i,x) for (auto i: x)
#define mem(x,i) memset(x,i,sizeof x)
#define fast ios_base::sync_with_stdio(false),cin.tie(0)

int N,M;
int ans[10];

int main(){
	fast;
	
	cin>>N>>M;
	
	mem(ans,-1);
	
	if (N == 1 && M == 0){
		cout<<0;
		return 0;
	}
	FOR(i,0,M-1){
		int s,c;
		cin>>s>>c;
		if (ans[s] != -1 && ans[s] != c){
			cout<<-1;
			return 0;
		}
		
		if (s == 1 && c == 0 && N > 1){
			cout<<-1;
			return 0;
		}
		
		ans[s] = c;
	}
	
	if (ans[1] == -1) ans[1] = 1;
	

	FOR(i,1,N){
		if (ans[i] == -1) cout<<0;
		else cout<<ans[i];
	}
	
}


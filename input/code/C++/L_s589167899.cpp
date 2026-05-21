#include <bits/stdc++.h>
#include <string>
#include <math.h>
#include <sstream>
using namespace std;

#define endl "\n"
#define fastio ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
//STANDARD DATA TYPES
#define ll long long
#define llu unsigned long long
#define ld long double
//ADV DATA TYPES
#define pii pair < int , int >
#define pll pair < ll, ll >
#define mp make_pair
#define ff first
#define ss second
//C 9/++ DS
#define vi vector < int >
#define vll vector < ll >
#define pb push_back
#define pf push_front
#define iter :: iterator
#define all(x) x.begin(),x.end()
 
//priority_queue < pii, vector < pii >, greater < pii > > pq;
//lower_bound(v.begin(),v.end(),20);//for any sorted container
//CONSTANTS
#define MOD 998244353
#define INF 0x3f3f3f3f
#define MAXN (int)(2e4+20)

//#define for(int i=1; i<=n ; i++) for(int i=0; i<n; i++)
#define forl(i,n) for(ll i=0; i<n; i++)
//MOD OPERATIONS
inline ll fpow(ll n, ll k, int p = MOD) {ll r = 1; for (; k; k >>= 1) {if (k & 1) r = r * n % p; n = n * n % p;} return r;}
inline ll inv(ll a, ll p = MOD) {return fpow(a, p - 2, p);}
inline ll addmod(ll a, ll val, ll p = MOD) {{if ((a = (a + val)) >= p) a -= p;} return a;}
inline ll submod(ll a, ll val, ll p = MOD) {{if ((a = (a - val)) < 0) a += p;}return a;}
inline ll mult(ll a, ll b, ll p = MOD) {return (ll) a * b % p;}
const  llu int_max = pow(2,32) - 1 ;

bool compare(const pair<int, int>&i, const pair<int, int>&j)
{
    return i.ff < j.ff;
}

int main(){

	fastio
	//total - a[i] % 2 == 0
	int n;
	cin>>n;
	bool a[n+1], val[n+1] = {0};
	//int total[n] = {0} ;
	for(int i=1; i<=n; i++){
		cin>>a[i];
		if(a[i] == 1)
			val[i] = 1;

	}
	for(int i = n/2; i>=1 ; i--){
		int total = 0, ax = i;
		for(int j=i ; j<=n; j+=i){
			if(val[j])
				total++;
		}
		if((total-a[i]) % 2 == 0)
			continue;
		else{
			if(val[i] == 0){
				val[i] = 1;
			}
			else{
				val[i] = 0;
			}
		}
	}
	int m=0;
	for(int i=1; i<=n ; i++){
		if(val[i])
			m++;
	}
	cout<<m<<endl;
	for(int i=1; i<=n ; i++)
		if(val[i])
			cout<<i<<" ";

	return 0 ;
	
}


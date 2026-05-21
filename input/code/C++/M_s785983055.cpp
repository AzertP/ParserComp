#include<bits/stdc++.h>
using namespace std;

const long long mod = (long long)(1e9 + 7);
vector<long long> fac(100001, 1);


long long mul(long long a, long long b){
	return ((a%mod)*(b%mod))%mod;
}

void fact(){
	for(long long i=1;i<=100000;i++) fac[i] = mul(fac[i-1], i);
}

long long add(long long a, long long b){
	return (a%mod+b%mod)%mod;
}

long long sub(long long a, long long b){
	return (a%mod-b%mod)%mod;
}
long long pw(long long a, long long b){
	//base case
	if(b == 0) return 1; 
	if(b == 1) return a;
	long long cmp = pw(a, b/2);
	if(b%2) return mul(mul(cmp, a), cmp);
	else return mul(cmp, cmp);
}

long long mulc_inv(long long num){
	return pw(num, mod-2);
}

long long comb(long long n, long long k){
	if(n < k) return 0;
	else{
		long long ans = 1;
		ans = mul(fac[n], mulc_inv(mul(fac[k],fac[n-k])));
		return ans;
	}
}

int main(){
	ios_base::sync_with_stdio(0); cin.tie(0);
	int n,k; cin >> n >> k;
	vector<long long> a(n);
	for(auto &x:a) cin >> x;
	sort(a.rbegin(), a.rend());
	long long ans = 0;
	fact();
	for(int i=0;i<n;i++)
	 ans = add(ans, sub(mul(a[i],comb(n-i-1, k-1)),mul(a[n-i-1], comb(n-i-1, k-1))));
	if(ans < 0) ans += mod;
	cout << ans << "\n";
	//cout << mul(mulc_inv(100),100) << "\n";
	//cout << comb(34738,3434) << "\n";
	//cout << mul(fac[], mulc_inv(k))
	//cout << fac[5] << "\n";
	return 0;
}

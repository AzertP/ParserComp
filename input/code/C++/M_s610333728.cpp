
using namespace std;

typedef long long ll;
typedef long double ld;
typedef unsigned long long ull;
const ll oo = 1e18;
const int mod = 1e9+7;
const int maxn = 3030;

ll dp[maxn][maxn];

int main(){
	int n;
	string s;
	cin>>n>>s;
	dp[1][1] = 1;
	for(int i=2;i<=n;i++){
		ll cur = 0;
		if(s[i-2] == '<'){
			for(int j=1;j<=i;j++){
				dp[j][i] = (dp[j][i] + cur) % mod;
				cur = (cur + dp[j][i-1]) % mod;
			}
		}else{
			for(int j=i;j>=1;j--){
				cur = (cur + dp[j][i-1]) % mod;
				dp[j][i] = (dp[j][i] + cur) % mod;
			}
		}
	}
	ll ans = 0;
	for(int i=1;i<=n;i++)
		ans = (ans + dp[i][n]) % mod;
	cout << ans << endl;
}

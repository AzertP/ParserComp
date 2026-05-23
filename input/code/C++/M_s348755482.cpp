using namespace std;
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

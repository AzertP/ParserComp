

const int MOD = 1000000007, INF = 1111111111; 
using namespace std;
typedef long long lint;

int main() {

	cin.tie(0);
	ios::sync_with_stdio(false);

	string s, t;
	cin >> s >> t;

	vector<vector<int>> dp(s.length() + 1, vector<int>(t.length() + 1));
	int s_len = (int)s.length(), t_len = (int)t.length();
	for (int i = 1; i <= s_len; i++) {
		for (int j = 1; j <= t_len; j++) {

			if (s[i - 1] == t[j - 1] && dp[i][j] < dp[i - 1][j - 1] + 1) dp[i][j] = dp[i - 1][j - 1] + 1;
			else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);

		}
	}
	
	string ans;
	int x = (int)s.length(), y = (int)t.length();
	while (x > 0 && y > 0) {

		if (dp[x][y] == dp[x - 1][y]) x--;
		else if (dp[x][y] == dp[x][y - 1]) y--;
		else {
			ans.push_back(s[x - 1]);
			x--, y--;
		}
	}

	reverse(ans.begin(), ans.end());
	cout << ans << "\n";

	return 0;

}

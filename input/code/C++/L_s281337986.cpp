
typedef long long ll;

using namespace std;

template<class T> inline bool chmax(T& a, T b) { if (a < b) { a = b; return 1; } return 0; }
template<class T> inline bool chmin(T& a, T b) { if (a > b) { a = b; return 1; } return 0; }

const long long INF = 1000000000000000;
const ll inf = -1e18;
typedef pair<ll, ll> P;
ll ma = 1000000000 + 7;
ll mx = 1000003;
ll h,n, w,m,t; string s;
int dx[4] = { 1, 0, -1, 0 };
int dy[4] = { 0, 1, 0, -1 };
ll gcd(ll x, ll y) {
	if (x % y == 0) return y;
	return gcd(y, x % y);
}
ll lcm(ll a,ll b) {
	ll g = gcd(a, b);
	return a / g * b;
}
void comb(vector<vector <ll> >& v) {
	for (ll i = 0; i < v.size(); i++) {
		v[i][0] = 1;
		v[i][i] = 1;
	}
	for (ll k = 1; k < v.size(); k++) {
		for (int j = 1; j < k; j++) {
			v[k][j] = (v[k - 1][j - 1] + v[k - 1][j])%ma;
		}
	}
}

ll GetDigit(ll num) {
	return log10(num) + 1;
}
ll Combination(int n, int r)
{
	if (n - r < r) r = n - r;
	if (r == 0) return 1;
	if (r == 1) return n;

	vector<ll> numerator(r);
	vector<ll> denominator(r);

	for (int k = 0; k < r; k++)
	{
		numerator[k] = n - r + k + 1;
		denominator[k] = k + 1;
	}

	for (int p = 2; p <= r; p++)
	{
		int pivot = denominator[p - 1];
		if (pivot > 1)
		{
			int offset = (n - r) % p;
			for (int k = p - 1; k < r; k += p)
			{
				numerator[k - offset] /= pivot;
				denominator[k] /= pivot;
			}
		}
	}

	ll result = 1;
	for (int k = 0; k < r; k++)
	{
		if (numerator[k] > 1) result *= numerator[k];
		result %= ma;
	}

	return result;
}
int main() {
	cin >> n;
	if (n % 2 == 1) {
		cout << 0 << endl;
		return 0;
	}

	ll res = n / 10;
	for (ll i = 50;; i *= 5) {
		if (i > n)break;
		res += n / i;
	}
	cout << res << endl;
}

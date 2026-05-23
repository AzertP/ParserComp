typedef long long ll;
using namespace std;
int main() {
	ll x, y;
	cin >> x >> y;
	ll ans = 0;
	while(x <= y) {
		x *= 2;
		ans ++;
	}
	cout << ans << endl;
}

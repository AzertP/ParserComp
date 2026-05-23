using namespace std;

int n, k, r, len;
set <int> s;

int main() {
	ios::sync_with_stdio(0);
	n = 7;
	cin >> k;
	r = n % k;
	len = 1;
	while(r != 0) {
		if(s.find(r) != s.end()) {
			cout << -1 << endl;
			return 0;
		}
		s.insert(r);
		n = r * 10 + 7;
		r = n % k;
		len++;
	}
	cout << len << endl;
	return 0;
}

using namespace std;
typedef long long int ll;
int Min(int a, int b, int c) {
	if (a <= b) { return std::min(a, c); }
	else return std::min(b, c);
}
int main(void) {
	char a, b; cin >> a >> b;
	if (a == b)cout << "=" << endl;
	else cout << (a > b ? ">" : "<") << endl;
	return 0;
}

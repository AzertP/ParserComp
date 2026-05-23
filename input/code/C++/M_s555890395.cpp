//-> tolower(),toupper()
//int a = stoi(c); int
//
// double
//map<,> p p[] = 
//mapfor (auto i = p.begin(); i != p.end();i++){}
//map keyp->first valuep->second  
using namespace std;

// N
ll dig(ll N) {
	ll dig = 0;
	while (N) {
		dig++;
		N /= 10;
	}
	return dig;
}
// x,y
ll gcd(ll x, ll y) {
	ll r;
	while (x%y) {
		r = x % y;
		x = y;
		y = r;
	}
	return y;
}   // 


int main() {
	double L; cin >> L;
	cout << setprecision(20) << pow(L, 3) / 27 << endl;
}

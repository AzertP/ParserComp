//大文字->小文字 tolower(),逆はtoupper()
//int a = stoi(c); 文字列をintへ
//途中の出力をそのまま残さない
//数値計算 基本はdouble
//map<キー,値> p は辞書。p[キー] = 値
//map全探索　for (auto i = p.begin(); i != p.end();i++){}
//mapの要素 keyはp->first valueはp->second  
#include <bits/stdc++.h>
#define rep(i,N) for(int i = 0; i < N;i++)
#define ALL(a) (a).begin(),(a).end()
#define ll long long int
using namespace std;

// Nの桁数
ll dig(ll N) {
	ll dig = 0;
	while (N) {
		dig++;
		N /= 10;
	}
	return dig;
}
// x,yの最大公約数
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
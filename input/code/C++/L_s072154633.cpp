#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <set>
#include <string>
#include <math.h>
#include <map>
#include <climits>
#include <iomanip>


#define forn(i, n) for (int i = 0; i < int(n); i++)

#define ll long long

using namespace std;

void A();
void B();
void C();
void D();
void E();
void F();

int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	B();


}

void A() {

	int K, A, B;
	cin >> K >> A >> B;

	double qA = (double)A / (double)K;
	double qB = (double)B / (double)K;

	cout << (qA <= (int)qB ? "OK" : "NG");


}

void B() {

	ll Bal = 100;

	ll Goal;
	cin >> Goal;

	int yr = 0;
	while (Bal < Goal) {

		Bal += Bal / 100;
		yr++;
	}

	cout << yr;

}

void C() {

}

void D() {
	int T;
	cin >> T;
	for (int t = 1; t <= T; t++) {

		int N;
		cin >> N;


		int result = 0;

		cout << "Case #" << t << ": ";

		cout << result << endl;

	}
}

void E() {
	int T;
	cin >> T;
	for (int t = 1; t <= T; t++) {

		int N;
		cin >> N;


		int result = 0;

		cout << "Case #" << t << ": ";

		cout << result << endl;

	}
}

void F() {
	int T;
	cin >> T;
	for (int t = 1; t <= T; t++) {

		int N;
		cin >> N;


		int result = 0;

		cout << "Case #" << t << ": ";

		cout << result << endl;

	}
}


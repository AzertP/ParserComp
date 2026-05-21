#include <iostream>
using namespace std;

int main() {
	int n;
	cin >> n;
	int a, c;
	a = n / 100;
	c = (n % 100) % 10;
	if (a == c) {
		cout << "Yes";
	}
	else {
		cout << "No";
	}

	return 0;
}
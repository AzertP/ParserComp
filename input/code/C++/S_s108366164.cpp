#include<iostream>
using namespace std;

int main() {
	int x1, y1, x2, y2; cin >> x1 >> y1 >> x2 >> y2;
	int x = x2;
	int y = y2;
	int dx = x2 - x1;
	int dy = y2 - y1;
	for (int i = 0; i < 2; i++) {
		swap(dx, dy);
		dx = -dx;
		x += dx;
		y += dy;
		cout << x << ' ' << y << ' ';
	}
	cout << endl;

	return 0;
}
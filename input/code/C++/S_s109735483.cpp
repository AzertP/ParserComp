#include <bits/stdc++.h>

using namespace std;

int read() {
	int x = 0; char c = getchar();
	while (c > '9' || c < '0') c = getchar();
	while (c >= '0' && c <= '9') x = x * 10 + c - 48, c = getchar();
	return x;
}

const int maxn = 1e5 + 10;

long long d[maxn], a[maxn], n, sum, t;

int main() {
	n = read();
	for (int i = 0; i < n; i ++) {
		a[i] = read();
		sum += a[i];
		t += (i + 1);
	}
	if (sum % t) puts("NO");
	else {
		t = sum / t;
		for (int i = 0; i < n; i ++) d[i] = a[(i + 1) % n] - a[i] - t;
		for (int i = 0; i < n; i ++) if (d[i] % n || d[i] > 0) return puts("NO"), 0; 
		puts("YES");
	}
	return 0;
}
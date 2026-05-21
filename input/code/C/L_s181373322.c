#include <stdio.h>
#include <math.h> //=> pow
#define NEWTON_LIMIT 10000
#define D 4

int abs(int val) {
	if (val < 0) { val *= -1; }
	return val;
}

double fabs(double val) {
	if (val < 0) { val *= -1; }
	return val;
}
/*
double pow(double val, int p) {
double ret = val;
for (int i = 1; i < p; i++) {
ret *= val;
}
return ret;
}
*/

//	CubicRoot by newton algorithm
double cbrt_newton(double a, double x) {
	double e;
	double a_new;
	int i = 0;

	while (i < NEWTON_LIMIT) {
		e = (x * x * x - a) / (3.0 * x * x);
		if (fabs(e) > 1.0e-16) { break; }
		x = x - e;
		i++;
	}
	return x - e;
}

double sqrt(double val) {
	double x = val / 2.0;
	double last_x = 0.0;

	while (x != last_x) {
		last_x = x;
		x = (x + val / x) / 2.0;
	}

	return x;
}

double max(double val[], int n) {
	double max = 0;
	int i = 0;
	for (; i < n; i++) {
		max = (max < val[i]) ? val[i] : max;
	}
	return max;
}

int main() {

	int n;
	int x[1000], y[1000];
	int *p[] = { x, y };
	double d[D];
	double xy[1000];
	int i, j;

	scanf("%d", &n);

	for (i = 0; i < 2; i++) {
		for (int j = 0; j < n; j++) {
			scanf("%d", &p[i][j]);
		}
	}

	for (j = 0; j < n; j++) {
		xy[j] = abs(x[j] - y[j]);
	}

	for (i = 0; i < D; i++) {
		d[i] = 0;

		if (i < 3) {
			for (j = 0; j < n; j++) {
				d[i] += pow(xy[j], i + 1);
			}
			//	すべてゼロだった場合の例外処理
			//	=> i=2...ニュートン法で三乗根求めるときにゼロ除算が入ってnan(ind)になるため．
			if (d[i] == 0) { continue; }
		}

			switch (i) {
		case 1:	d[i] = sqrt(d[i]); break;
		case 2: d[i] = cbrt_newton(d[i], pow(d[i], 1.0 / 3.0)); break;
		case 3: d[i] = max(xy, n); break;
		}
	}

	for (int i = 0; i < D; i++) {
		printf("%.8f\n", d[i]);
	}

	return 0;
}

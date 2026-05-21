#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
int P[100], T[100], V[100];
double prob[100][51];
double comb[51][51];
int main() {
	int N, M, L;
	cin >> N >> M >> L;
	comb[0][0] = 1;
	for (int i = 1; i <= M; i++) {
		comb[i][0] = comb[i][i] = 1;
		for (int j = 1; j < i; j++) {
			comb[i][j] = comb[i - 1][j - 1] + comb[i - 1][j];
		}
	}
	for (int i = 0; i < N; i++) {
		cin >> P[i] >> T[i] >> V[i];
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j <= M; j++) {
			prob[i][j] = comb[M][j] * pow(P[i] / 100.0, j)*pow(1 - P[i] / 100.0, M - j);
		}
	}
	for (int i = 0; i < N; i++) {
		if (V[i] == 0) {
			cout << 0 << endl;
			continue;
		}
		double ans = 0;
		for (int j = 0; j <= M; j++) {
			double t1 = (double)L / V[i] + T[i] * j;
			double p = 1;
			for (int k = 0; k < N; k++) {
				if (k == i) continue;
				if (V[k] == 0) continue;
				double sum = 0;
				for (int l = 0; l <= M; l++) {
					double t2 = (double)L / V[k] + T[k] * l;
					if (t1 < t2) sum += prob[k][l];
				}
				p *= sum;
			}
			ans += prob[i][j] * p;
		}
		printf("%.15lf\n", ans);
	}
}

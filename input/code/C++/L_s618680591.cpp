#include<bits/stdc++.h>
using namespace std;
 
int N, M; char S[15][15],T[15][15];
vector<pair<int, int> > pick;
int chk[15];
 
void go(int r)
{
	if (r <= 1) {
		int x = -1;
		for (int i = 0; i < N; i++) if (chk[i] == 2) x = i;
		int v = 0;
		for (int i = 0; i < pick.size(); i++) {
			for (int j = 0; j < M; j++) T[v][j] = S[pick[i].first][j];
			v++;
		}
		if (x >= 0) {
			for (int j = 0; j < M; j++) T[v][j] = S[x][j];
			v++;
		}
		for (int i = (int)pick.size() - 1; i >= 0; i--) {
			for (int j = 0; j < M; j++) T[v][j] = S[pick[i].second][j];
			v++;
		}
 
		vector<string> p;
		for (int j = 0; j < M; j++) {
			string u;
			for (int i = 0; i < N; i++) u += T[i][j];
			p.push_back(u);
		}
 
		for (int i = 0; i < p.size(); i++) {
			string v = p[i];
			reverse(v.begin(), v.end());
			for (int j = i + 1; j < p.size(); j++) {
				if (v == p[j]) {
					p.erase(p.begin() + j);
					p.erase(p.begin() + i);
					i--;
					break;
				}
			}
		}
 
		if (p.size() == 1) {
			string v = p[0];
			reverse(v.begin(), v.end());
			if (p[0] != v) return;
		}
 
		if (p.size() <= 1) {
			puts("YES");
			exit(0);
		}
	}
	else {
		int s;
		for (int i = 0; i < N; i++) if (!chk[i]) { s = i; break; }
		chk[s] = 1;
		for (int i = s + 1; i < N; i++) if (!chk[i]) {
			chk[i] = 1;
			pick.push_back({ s,i });
			go(r - 2);
			pick.pop_back();
			chk[i] = 0;
		}
		chk[s] = 0;
	}
}
 
void proc()
{
	scanf("%d %d", &N, &M);
	for (int i = 0; i < N; i++) scanf("%s", S[i]);
	if (N % 2) {
		for (int i = 0; i < N; i++) {
			chk[i] = 2;
			go(N - 1);
			chk[i] = 0;
		}
	}
	else go(N);
	puts("NO");
}
 
int main()
{
	proc();
	return 0;
}

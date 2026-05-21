#include<bits/stdc++.h>
using namespace std;
#define il inline
#define re register
#define int long long
#define D double
il int read() {
    re int x = 0, f = 1; re char c = getchar();
    while(c < '0' || c > '9') { if(c == '-') f = -1; c = getchar();}
    while(c >= '0' && c <= '9') x = x * 10 + c - 48, c = getchar();
    return x * f;
}
#define rep(i , a , b) for(int i = (a) , i##Limit = (b) ; i <= i##Limit ; ++ i)
#define maxn 200005
int n, m, a[maxn], ans, Mi = -1e18, Ma = 1e18, c1, c2, b[maxn], c[maxn], g, d[maxn], c3;
il int check(int x) {
	int tem = 0;
	if(x >= 0) {
		tem = g * (c1 + c2) + g * (g - 1) / 2 + c1 * c2;
		rep(i, 1, c1 - 1) tem += upper_bound(b + i + 1, b + c1 + 1, x / b[i]) - b - 1 - i;
		rep(i, 1, c2 - 1) tem += upper_bound(c + i + 1, c + c2 + 1, x / c[i]) - c - 1 - i;
	}
	else {
		x = -x;
		rep(i, 1, c1){
            int pos = lower_bound(c + 1, c + c2 + 1, (int)ceil((D)x / (D)b[i])) - c;
            if(b[i] * c[pos] < x) pos++;
            if(b[i] * c[pos - 1] >= x) -- pos;
            if(c2 >= pos) tem += c2 - pos + 1;
        } 
	}
	return tem >= m;
}
signed main() {
	n = read(), m = read();
	rep(i, 1, n) a[i] = read();
	rep(i, 1, n) {
		if(a[i] < 0) b[++ c1] = -a[i];
		else if(a[i] > 0) c[++ c2] = a[i];
		else ++ g;
	}
	sort(b + 1, b + c1 + 1), sort(c + 1, c + c2 + 1);
	while(Mi < Ma){
        int Mid = (Mi + Ma) >> 1;
        if(check(Mid)) Ma = Mid;
        else Mi = Mid + 1;
    }
	cout << Ma;
	return 0;
}
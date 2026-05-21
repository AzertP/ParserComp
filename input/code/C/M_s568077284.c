// AOJ 2442: ConvexCut
// 2017.11.17 bal4u@uu

#include <stdio.h>
#include <string.h>

int x[51], y[51];

long long norm(int dx, int dy)
{
	return (long long)dx*dx + (long long)dy*dy;
}

int parallel(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4)
{
	return (long long)(y2-y1)*(long long)(x4-x3) == (long long)(x2-x1)*(long long)(y4-y3);
}

int main()
{
	int n, n2, i, j;

	scanf("%d", &n);
	if (n & 1) goto na;
	n2 = n >> 1;
	for (i = 0; i < n; i++) scanf("%d%d", x+i, y+i);
	x[n] = x[0], y[n] = y[0];
	for (j = n2, i = 0; i < n2; i++, j++) {
		if (norm(x[i]-x[i+1], y[i]-y[i+1]) != norm(x[j]-x[j+1], y[j]-y[j+1])) goto na;
		if (!parallel(x[i], y[i], x[i+1], y[i+1], x[j], y[j], x[j+1], y[j+1])) goto na;
	}
	printf("%.10lf %.10lf\n", (x[0]+x[n2])/2.0, (y[0]+y[n2])/2.0);
	return 0;
na: puts("NA");
	return 0;
}
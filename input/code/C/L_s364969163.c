#include <stdio.h>          // printf(), scanf()
#include <stdbool.h>

#define N 8

int max, min;
int d[N];
bool flag[N];

void
permutation(int n, int lv)
{
	static int p[N];
	int memo[N];
	int a;
	int i, j;

	flag[n] = true;
	p[lv] = d[n];

	j = 0;
	for (i = 0; i < N; ++i)
	{
		if (!flag[i])
			memo[j++] = i;
	}

	if (j == 1)
	{
		p[lv + 1] = d[memo[0]];

		a = 0;
		for (i = 0; i < N; ++i)
			a = a * 10 + p[i];

		if (a < min)
			min = a;

		if (a > max)
			max = a;

		flag[n] = false;
		return;
	}

	for (i = 0; i < j; ++i)
		permutation(memo[i], lv + 1);

	flag[n] = false;
}

int
main(int argc, char **argv)
{
	int n;
	char c;
	int i;

	scanf("%d\n", &n);
	while (n--)
	{
		for (i = 0; i < N; ++i)
		{
			scanf("%c", &c);
			d[i] = c - '0';
			flag[i] = false;
		}

		min = 99999999;
		max = 0;
		for (i = 0; i < N; ++i)
			permutation(i, 0);

		scanf("%c", &c);
		printf("%d\n", max - min);
	}

	return 0;
}
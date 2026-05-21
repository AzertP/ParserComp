#include <stdio.h>

int main(void)
{
	int a[4][3][10] = {0};
	int n, i;
	int bl, fl, ro;

	scanf("%d", &n);

	for (i = 0; i < n; i++) {
		int b, f, r, v;
		scanf("%d %d %d %d", &b, &f, &r, &v);
		a[b - 1][f - 1][r - 1] += v;
	}

	for (bl = 1; bl < 5; bl++) {
		for ( fl = 1; fl < 4; fl++) {
			for ( ro = 1; ro < 11; ro++)
				printf(" %d", a[bl - 1][fl - 1][ro - 1]);
			putchar('\n');
		}
		if (bl < 4)
			printf("####################\n");
	}
	
    return 0;
}
#include <stdio.h>

int
main(void)
{
	int a, b;
	scanf("%d %d", &a, &b);

	int res;
	if (a > b)
		res = a - 1;
	else
		res = a;
	printf("%d\n", res);
	return (0);
}

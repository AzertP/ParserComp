#include <stdio.h>
#include <stdlib.h>

int compare_unsigned_int(const void *a, const void *b)
{
	return *(unsigned int*)b - *(unsigned int*)a;
}


int main() {

	int n;	
	int k;	

	scanf("%d %d", &n, &k);

	unsigned int h[n];
	int i;
	double attack_sum = 0.0;

	for(i = 0; i < n; i++) {
		scanf("%d", &h[i]);
	}

	qsort(h, n, sizeof(unsigned int), compare_unsigned_int);
	for(i = k; i < n; i++) {
		attack_sum += h[i];
	}
	printf("%.0f\n", attack_sum);

	return 0;
}

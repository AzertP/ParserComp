

int
main(int argc, const char *argv[])
{
	char buffer[BUFFER_LENGTH];
	int k;
	int a;	/* a biscuits to 1 yen */
	int b;	/* 1 yen to b biscuits */
	long int n;
	int up;
	int kk;
	int kkm;
	int kkk;

	(void)fgets(buffer, sizeof(buffer), stdin);
	(void)sscanf(buffer, "%d %d %d\n", &k, &a, &b);

	if (a >= b) {
		n = 1 + k;
	} else {
		up = -a + b;
		kk = k - a + 1;
		kkm = kk % 2;
		kkk = kk / 2;
		n = a + kkm + ((long int)kkk * up);
		if (n < (1 + k)) {
			n = 1 + k;
		}
	}

	(void)printf("%ld\n", n);

	return 0;
}

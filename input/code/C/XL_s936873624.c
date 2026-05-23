
// 

// 
static FILE *szpFpI;											// 
static int si1Array[D_ARRAY_MAX];								// 
static int siACnt;												// 
static int siSCnt;												// 
static int si1ModFact[D_FACT_MAX];								// 
static int si1ModFactR[D_FACT_MAX];								// ()

//  - 
	static int siRes;
	static FILE *szpFpA;
	static int siTNo;

// 
int
fOutLine(
	char *pcpLine				// <I> 
)
{
	char lc1Buf[1024];

	lc1Buf[0] = '\0';
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, pcpLine)) {
		siRes = -1;
	}
	printf("%s", pcpLine);

	return 0;
}

//  - int
int
fSortFncIU(
	const void *pzpVal1			// <I> 
	, const void *pzpVal2		// <I> 
)
{
	int *lipVal1 = (int *)pzpVal1;
	int *lipVal2 = (int *)pzpVal2;

	// int
	if (*lipVal1 > *lipVal2) {
		return 1;
	}
	else if (*lipVal1 < *lipVal2) {
		return -1;
	}

	return 0;
}

//  - 
int
fGetModPower(
	int piBase					// <I> 
	, int piIdx					// <I> 
)
{
	//  - 
	int li1Val[100];
	li1Val[0] = piBase;
	int liCnt = 1;
	int liIdx = 1;
	while (piIdx > liIdx) {
		li1Val[liCnt] = (int)((long long)li1Val[liCnt - 1] * (long long)li1Val[liCnt - 1] % D_MOD);
		liCnt++;
		liIdx += liIdx;
	}

	//  - 
	int liVal = 1;
	while (piIdx > 0) {
		if (piIdx >= liIdx) {
			piIdx -= liIdx;
			liVal = (int)((long long)liVal * (long long)li1Val[liCnt - 1] % D_MOD);
		}
		liCnt--;
		liIdx /= 2;
	}

	return liVal;
}

//  - 
int
fMakeModFact(
	int piMax					// <I> 
)
{
	int i;

	si1ModFact[0] = 1;
	si1ModFact[1] = 1;
	for (i = 2; i <= piMax; i++) {
		si1ModFact[i] = (int)((long long)si1ModFact[i - 1] * (long long)i % D_MOD);
	}

	return 0;
}

// () - 
int
fMakeModFactR(
	int piMax					// <I> 
)
{
	int i;

	for (i = 0; i <= piMax; i++) {
		si1ModFactR[i] = fGetModPower(si1ModFact[i], D_MOD - 2);
	}

	return 0;
}

// nCk - 
int
fGetnCk(
	int piN						// <I> N
	, int piK					// <I> K
)
{
	if (piN < piK) {
		return 0;
	}
	if (piK < 0) {
		return 0;
	}
	int liCnt = (int)((long long)si1ModFact[piN] * (long long)si1ModFactR[piN - piK] % D_MOD);
	return (int)((long long)liCnt * (long long)si1ModFactR[piK] % D_MOD);
}

// 
int
fMain(
)
{
	int i;
	char lc1Buf[1024];

	//  - 
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	sscanf(lc1Buf, "%d%d", &siACnt, &siSCnt);
	if (siACnt < 2) {
		return 0;
	}

	//  - 
	for (i = 0; i < siACnt; i++) {
		fscanf(szpFpI, "%d", &si1Array[i]);
	}
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	qsort(si1Array, siACnt, sizeof(int), fSortFncIU);

	//  - 
	fMakeModFact(siACnt);
	fMakeModFactR(siACnt);

	//  - 
	long long llMin = 0;
	long long llMax = 0;
	for (i = 0; i < siACnt; i++) {

		// 
		long long llnCk = fGetnCk(siACnt - i - 1, siSCnt - 1);
		llMin = (llMin + (long long)si1Array[i] * llnCk) % D_MOD;

		// 
		llnCk = fGetnCk(i, siSCnt - 1);
		llMax = (llMax + (long long)si1Array[i] * llnCk) % D_MOD;
	}

	return (int)((llMax - llMin + D_MOD) % D_MOD);
}

// 
int
fOne(
)
{
	int liRet;
	char lc1Buf[1024];

	//  - 
	sprintf(lc1Buf, ".\\Test\\T%d.txt", siTNo);
	szpFpI = fopen(lc1Buf, "r");
	sprintf(lc1Buf, ".\\Test\\A%d.txt", siTNo);
	szpFpA = fopen(lc1Buf, "r");
	siRes = 0;
	szpFpI = stdin;

	// 
	liRet = fMain();

	// 
	sprintf(lc1Buf, "%d\n", liRet);
	fOutLine(lc1Buf);

	// 
	lc1Buf[0] = '\0';
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, "")) {
		siRes = -1;
	}

	// 
	fclose(szpFpI);
	fclose(szpFpA);

	// 
	if (siRes == 0) {
		printf("OK %d\n", siTNo);
	}
	else {
		printf("NG %d\n", siTNo);
	}

	return 0;
}

// 
int
main()
{

	int i;
	for (i = D_TEST_SNO; i <= D_TEST_ENO; i++) {
		siTNo = i;
		fOne();
	}
	fOne();

	return 0;
}


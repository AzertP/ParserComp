
// 

// 
static FILE *szpFpI;											// 
static int si1Heap[1000];									// 
static int siHCnt;												// 

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

	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, pcpLine)) {
		siRes = -1;
	}
	printf("%s", pcpLine);

	return 0;
}

//  -  - 
int
fHeapCmp(
	int piNo1					// <I>  0
	, int piNo2					// <I>  0
)
{
	// 
	if (si1Heap[piNo1] < si1Heap[piNo2]) {
		return 1;
	}
	else if (si1Heap[piNo1] > si1Heap[piNo2]) {
		return -1;
	}

	return 0;
}

//  - 
// [>=0]: [-1]:
int
fHeapChk(
	int piPNo					// <I>  0
)
{
	int liRet;

	// 
	int liMNo = piPNo;

	// 
	int liCNo = piPNo * 2 + 1;
	if (liCNo < siHCnt) {
		liRet = fHeapCmp(liMNo, liCNo);
		if (liRet == 1) {
			liMNo = liCNo;
		}
	}

	// 
	liCNo = piPNo * 2 + 2;
	if (liCNo < siHCnt) {
		liRet = fHeapCmp(liMNo, liCNo);
		if (liRet == 1) {
			liMNo = liCNo;
		}
	}

	// 
	if (piPNo == liMNo) {
		return -1;
	}

	// 
	int liWork;
	liWork = si1Heap[liMNo];
	si1Heap[liMNo] = si1Heap[piPNo];
	si1Heap[piPNo] = liWork;

	return liMNo;
}

//  - 
int
fHeapEnqueue(
	int piVal					// <I> 
)
{
	int liRet;

	// 
	si1Heap[siHCnt] = piVal;
	siHCnt++;

	// 
	int liNo = siHCnt - 1;
	while (1) {

		// 
		liNo = (liNo - 1) / 2;

		// 
		liRet = fHeapChk(liNo);
		if (liRet < 0) {
			break;
		}
	}

	return 0;
}

//  - 
int
fHeapDequeue(
	int *pipRet					// <O> 
)
{
	// 
	if (siHCnt < 1) {
		return -1;
	}

	// 
	*pipRet = si1Heap[0];
	siHCnt--;

	// 
	if (siHCnt < 1) {
		return 0;
	}

	// 
	si1Heap[0] = si1Heap[siHCnt];

	// 
	int liNo = 0;
	while (liNo >= 0) {
		liNo = fHeapChk(liNo);
	}

	return 0;
}

// 
long long
fMain(
)
{
	int i;
	char lc1Buf[1024];

	//  - 
	int liICnt, liDCnt;
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	sscanf(lc1Buf, "%d%d", &liICnt, &liDCnt);

	//  - 
	int liVal;
	for (i = 0; i < liICnt; i++) {
		fscanf(szpFpI, "%d", &liVal);
		fHeapEnqueue(liVal);
	}
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);

	//  - 
	for (i = 0; i < liDCnt; i++) {
		fHeapDequeue(&liVal);
		fHeapEnqueue(liVal / 2);
	}

	//  - 
	long long llSum = 0;
	for (i = 0; i < liICnt; i++) {
		fHeapDequeue(&liVal);
		llSum += liVal;
	}

	return llSum;
}

// 
int
fOne(
)
{
	long long llRet;
	char lc1Buf[1024];

	//  - 
	siHCnt = 0;													// 

	//  - 
	sprintf(lc1Buf, ".\\Test\\T%d.txt", siTNo);
	szpFpI = fopen(lc1Buf, "r");
	sprintf(lc1Buf, ".\\Test\\A%d.txt", siTNo);
	szpFpA = fopen(lc1Buf, "r");
	siRes = 0;
	szpFpI = stdin;

	// 
	llRet = fMain();

	// 
	sprintf(lc1Buf, "%lld\n", llRet);
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


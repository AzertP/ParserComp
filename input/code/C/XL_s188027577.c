
// 

//  - 
typedef struct Tree {
	int miSum;																// 
	long long mlCnt;														// 
	int mi1Height[D_TREE_WCNT];												// 
	struct Tree *mzp1Child[D_TREE_WCNT];									// 
} Tree;

// 
static Tree sz2Tree[D_TREE_CNT][D_CARD_VAL * D_CARD_MAX];					// 
static int si1TCnt[D_TREE_CNT];												// 
static Tree *szp1Top[D_TREE_CNT];											// 

//  - 
	static int siRes;
	static FILE *szpFpT, *szpFpA;

//  - 
int
fTreeClear(
	int piTNo					// <I>  0
)
{
	si1TCnt[piTNo] = 0;			// 
	szp1Top[piTNo] = NULL;		// 

	return 0;
}

//  - 
Tree *
fTreeMake(
	int piTNo					// <I>  0
	, int piSum					// <I> 
	, long long plCnt			// <I> 
)
{
	// 
	Tree *lzpTree = &(sz2Tree[piTNo][si1TCnt[piTNo]]);
	(si1TCnt[piTNo])++;

	// 
	memset(lzpTree, 0, sizeof(Tree));		// 
	lzpTree->miSum = piSum;					// 
	lzpTree->mlCnt = plCnt;					// 

	return lzpTree;
}

//  - 
int
fTreeCmp(
	int piTNo					// <I>  0
	, int piSum					// <I> 
	, Tree *pzpTree				// <I> 
)
{
	// 
	if (piSum < pzpTree->miSum) {
		return -1;
	}
	else if (piSum > pzpTree->miSum) {
		return 1;
	}

	return 0;
}

//  - 
int
fTreeGetHeight(
	Tree *pzpTree				// <I> 
)
{
	// 
	if (pzpTree == NULL) {
		return 0;
	}

	if (pzpTree->mi1Height[D_TREE_LEFT] >= pzpTree->mi1Height[D_TREE_RIGHT]) {
		return pzpTree->mi1Height[D_TREE_LEFT] + 1;
	}
	else {
		return pzpTree->mi1Height[D_TREE_RIGHT] + 1;
	}
}

//  - ()
int
fTreeRttR(
	Tree **pzppTree				// <I> 
)
{
	// 
	Tree *lzpChild = (*pzppTree)->mzp1Child[D_TREE_LEFT];

	// 
	(*pzppTree)->mzp1Child[D_TREE_LEFT] = lzpChild->mzp1Child[D_TREE_RIGHT];	//  = 
	(*pzppTree)->mi1Height[D_TREE_LEFT] = lzpChild->mi1Height[D_TREE_RIGHT];	// () = ()
	lzpChild->mzp1Child[D_TREE_RIGHT] = *pzppTree;								//  = 
	lzpChild->mi1Height[D_TREE_RIGHT] = fTreeGetHeight(*pzppTree);				// () - 
	*pzppTree = lzpChild;														//  = 
	
	return 0;
}

//  - ()
int
fTreeRttL(
	Tree **pzppTree				// <I> 
)
{
	// 
	Tree *lzpChild = (*pzppTree)->mzp1Child[D_TREE_RIGHT];

	// 
	(*pzppTree)->mzp1Child[D_TREE_RIGHT] = lzpChild->mzp1Child[D_TREE_LEFT];	//  = 
	(*pzppTree)->mi1Height[D_TREE_RIGHT] = lzpChild->mi1Height[D_TREE_LEFT];	// () = ()
	lzpChild->mzp1Child[D_TREE_LEFT] = *pzppTree;								//  = 
	lzpChild->mi1Height[D_TREE_LEFT] = fTreeGetHeight(*pzppTree);				// () - 
	*pzppTree = lzpChild;														//  = 

	return 0;
}

//  - 
// [1] [0]
int
fTreeComAddDel(
	Tree **pzppNow				// <I> 
	, int piWay					// <I> 
)
{
	// 
	int liNew = fTreeGetHeight((*pzppNow)->mzp1Child[piWay]);
	if ((*pzppNow)->mi1Height[piWay] == liNew) {												// 
		return 0;
	}
	(*pzppNow)->mi1Height[piWay] = liNew;														// 

	// 
	if ((*pzppNow)->mi1Height[D_TREE_LEFT] - (*pzppNow)->mi1Height[D_TREE_RIGHT] > 1) {			// 
		fTreeRttR(pzppNow);																			// 
	}
	else if ((*pzppNow)->mi1Height[D_TREE_RIGHT] - (*pzppNow)->mi1Height[D_TREE_LEFT] > 1) {	// 
		fTreeRttL(pzppNow);																			// 
	}

	return 1;
}

//  - 
// [1] [0]
int
fTreeAdd(
	int piTNo					// <I>  0
	, Tree **pzppNow			// <I> 
	, int piSum					// <I> 
	, long long plCnt			// <I> 
)
{
	// 
	if (*pzppNow == NULL) {
		*pzppNow = fTreeMake(piTNo, piSum, plCnt);
		return 1;
	}

	// 
	int liRet = fTreeCmp(piTNo, piSum, *pzppNow);
	if (liRet == 0) {																	// 
		(*pzppNow)->mlCnt += plCnt;															// 
		return 0;
	}

	// 
	int liWay;
	if (liRet < 0) {																	// 
		liWay = D_TREE_LEFT;
	}
	else {																				// 
		liWay = D_TREE_RIGHT;
	}

	// 
	liRet = fTreeAdd(piTNo, &((*pzppNow)->mzp1Child[liWay]), piSum, plCnt);
	if (liRet == 0) {																	// 
		return 0;
	}

	// 
	return fTreeComAddDel(pzppNow, liWay);
}

// 
int
fMain(
	int piTNo					// <I>  1
)
{
	int i, j, k;
	char lc1Buf[1024], lc1Out[1024];

	// 
	sprintf(lc1Buf, ".\\Test\\T%d.txt", piTNo);
	szpFpT = fopen(lc1Buf, "r");
	sprintf(lc1Buf, ".\\Test\\A%d.txt", piTNo);
	szpFpA = fopen(lc1Buf, "r");
	siRes = 0;

	// 
	int liCCnt, liAvg;
	fgets(lc1Buf, sizeof(lc1Buf), szpFpT);
	fgets(lc1Buf, sizeof(lc1Buf), stdin);
	sscanf(lc1Buf, "%d%d", &liCCnt, &liAvg);

	// 
	int liFNo = D_TNO_A;
	int liTNo = D_TNO_B;
	fTreeClear(liFNo);								// 
	fTreeAdd(liFNo, &szp1Top[liFNo], 0, 1);			// 

	// 
	for (i = 0; i < liCCnt; i++) {
		int liCard;
		fscanf(szpFpT, "%d", &liCard);
		fscanf(stdin, "%d", &liCard);
		liCard -= liAvg;

		// ()
		fTreeClear(liTNo);

		// ()
		for (j = 0; j < si1TCnt[liFNo]; j++) {
			Tree *lzpTree = &sz2Tree[liFNo][j];

			// ()
			int li1Sum[2];
			li1Sum[0] = lzpTree->miSum;
			li1Sum[1] = li1Sum[0] + liCard;

			// 
			for (k = 0; k < 2; k++) {
				fTreeAdd(liTNo, &szp1Top[liTNo], li1Sum[k], lzpTree->mlCnt);
			}
		}

		// 
		if (liFNo == D_TNO_A) {
			liFNo = D_TNO_B;
			liTNo = D_TNO_A;
		}
		else {
			liFNo = D_TNO_A;
			liTNo = D_TNO_B;
		}
	}
	fgets(lc1Buf, sizeof(lc1Buf), szpFpT);
	fgets(lc1Buf, sizeof(lc1Buf), stdin);

	// 
	long long llSum = 0;
	for (i = 0; i < si1TCnt[liFNo]; i++) {
		Tree *lzpTree = &sz2Tree[liFNo][i];

		// 
		if (lzpTree->miSum != 0) {
			continue;
		}

		// 
		llSum += lzpTree->mlCnt;
	}

	// 
	sprintf(lc1Out, "%lld\n", llSum - 1);

	// 
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, lc1Out)) {
		siRes = -1;
	}
	printf("%s", lc1Out);

	// 
	fclose(szpFpT);
	fclose(szpFpA);

	// 
	if (siRes == 0) {
		printf("OK %d\n", piTNo);
	}
	else {
		printf("NG %d\n", piTNo);
	}

	return 0;
}

int
main()
{

	int i;
	for (i = D_TEST_SNO; i <= D_TEST_ENO; i++) {
		fMain(i);
	}
	fMain(0);

	return 0;
}


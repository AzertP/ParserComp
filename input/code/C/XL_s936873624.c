#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 内部定数
#define D_MOD			1000000007								// 除数(10の9乗+7)
#define D_ARRAY_MAX		100000									// 最大配列数
#define D_FACT_MAX		100005									// 最大階乗数

// 内部変数
static FILE *szpFpI;											// 入力
static int si1Array[D_ARRAY_MAX];								// 配列
static int siACnt;												// 配列数
static int siSCnt;												// 選択数
static int si1ModFact[D_FACT_MAX];								// 階乗リスト
static int si1ModFactR[D_FACT_MAX];								// 階乗リスト(逆元)

// 内部変数 - テスト用
#ifdef D_TEST
	static int siRes;
	static FILE *szpFpA;
	static int siTNo;
#endif

// １行出力
int
fOutLine(
	char *pcpLine				// <I> １行
)
{
	char lc1Buf[1024];

#ifdef D_TEST
	lc1Buf[0] = '\0';
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, pcpLine)) {
		siRes = -1;
	}
#else
	printf("%s", pcpLine);
#endif

	return 0;
}

// ソート関数 - int昇順
int
fSortFncIU(
	const void *pzpVal1			// <I> 値１
	, const void *pzpVal2		// <I> 値２
)
{
	int *lipVal1 = (int *)pzpVal1;
	int *lipVal2 = (int *)pzpVal2;

	// int昇順
	if (*lipVal1 > *lipVal2) {
		return 1;
	}
	else if (*lipVal1 < *lipVal2) {
		return -1;
	}

	return 0;
}

// べき乗 - 取得
int
fGetModPower(
	int piBase					// <I> 基数
	, int piIdx					// <I> 指数
)
{
	// 平方リスト - 作成
	int li1Val[100];
	li1Val[0] = piBase;
	int liCnt = 1;
	int liIdx = 1;
	while (piIdx > liIdx) {
		li1Val[liCnt] = (int)((long long)li1Val[liCnt - 1] * (long long)li1Val[liCnt - 1] % D_MOD);
		liCnt++;
		liIdx += liIdx;
	}

	// べき乗 - 取得
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

// 階乗リスト - 作成
int
fMakeModFact(
	int piMax					// <I> 最大値
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

// 階乗リスト(逆元) - 作成
int
fMakeModFactR(
	int piMax					// <I> 最大値
)
{
	int i;

	for (i = 0; i <= piMax; i++) {
		si1ModFactR[i] = fGetModPower(si1ModFact[i], D_MOD - 2);
	}

	return 0;
}

// nCk - 取得
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

// 実行メイン
int
fMain(
)
{
	int i;
	char lc1Buf[1024];

	// 配列数・選択数 - 取得
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	sscanf(lc1Buf, "%d%d", &siACnt, &siSCnt);
	if (siACnt < 2) {
		return 0;
	}

	// 配列 - 取得
	for (i = 0; i < siACnt; i++) {
		fscanf(szpFpI, "%d", &si1Array[i]);
	}
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	qsort(si1Array, siACnt, sizeof(int), fSortFncIU);

	// 階乗リスト - 作成
	fMakeModFact(siACnt);
	fMakeModFactR(siACnt);

	// 合計 - 取得
	long long llMin = 0;
	long long llMax = 0;
	for (i = 0; i < siACnt; i++) {

		// 最小値合計
		long long llnCk = fGetnCk(siACnt - i - 1, siSCnt - 1);
		llMin = (llMin + (long long)si1Array[i] * llnCk) % D_MOD;

		// 最大値合計
		llnCk = fGetnCk(i, siSCnt - 1);
		llMax = (llMax + (long long)si1Array[i] * llnCk) % D_MOD;
	}

	return (int)((llMax - llMin + D_MOD) % D_MOD);
}

// １回実行
int
fOne(
)
{
	int liRet;
	char lc1Buf[1024];

	// 入力 - セット
#ifdef D_TEST
	sprintf(lc1Buf, ".\\Test\\T%d.txt", siTNo);
	szpFpI = fopen(lc1Buf, "r");
	sprintf(lc1Buf, ".\\Test\\A%d.txt", siTNo);
	szpFpA = fopen(lc1Buf, "r");
	siRes = 0;
#else
	szpFpI = stdin;
#endif

	// 実行メイン
	liRet = fMain();

	// 出力
	sprintf(lc1Buf, "%d\n", liRet);
	fOutLine(lc1Buf);

	// 残データ有無
#ifdef D_TEST
	lc1Buf[0] = '\0';
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, "")) {
		siRes = -1;
	}
#endif

	// テストファイルクローズ
#ifdef D_TEST
	fclose(szpFpI);
	fclose(szpFpA);
#endif

	// テスト結果
#ifdef D_TEST
	if (siRes == 0) {
		printf("OK %d\n", siTNo);
	}
	else {
		printf("NG %d\n", siTNo);
	}
#endif

	return 0;
}

// プログラム開始
int
main()
{

#ifdef D_TEST
	int i;
	for (i = D_TEST_SNO; i <= D_TEST_ENO; i++) {
		siTNo = i;
		fOne();
	}
#else
	fOne();
#endif

	return 0;
}


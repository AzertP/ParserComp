#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 内部定数
#define D_HEAP_MAX		100000									// 最大ヒープ数

// 内部変数
static FILE *szpFpI;											// 入力
static int si1Heap[D_HEAP_MAX];									// ヒープ
static int siHCnt;												// ヒープ数

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
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, pcpLine)) {
		siRes = -1;
	}
#else
	printf("%s", pcpLine);
#endif

	return 0;
}

// ヒープ - 比較 - 降順
int
fHeapCmp(
	int piNo1					// <I> 配列番号１ 0～
	, int piNo2					// <I> 配列番号２ 0～
)
{
	// 降順
	if (si1Heap[piNo1] < si1Heap[piNo2]) {
		return 1;
	}
	else if (si1Heap[piNo1] > si1Heap[piNo2]) {
		return -1;
	}

	return 0;
}

// ヒープ - 親子関係チェック
// 戻り値：[>=0]:変更した子の配列番号 [-1]:変更なし
int
fHeapChk(
	int piPNo					// <I> 親の配列番号 0～
)
{
	int liRet;

	// 最小値
	int liMNo = piPNo;

	// 左の子と比較
	int liCNo = piPNo * 2 + 1;
	if (liCNo < siHCnt) {
		liRet = fHeapCmp(liMNo, liCNo);
		if (liRet == 1) {
			liMNo = liCNo;
		}
	}

	// 右の子と比較
	liCNo = piPNo * 2 + 2;
	if (liCNo < siHCnt) {
		liRet = fHeapCmp(liMNo, liCNo);
		if (liRet == 1) {
			liMNo = liCNo;
		}
	}

	// 変更有無
	if (piPNo == liMNo) {
		return -1;
	}

	// 値の交換
	int liWork;
	liWork = si1Heap[liMNo];
	si1Heap[liMNo] = si1Heap[piPNo];
	si1Heap[piPNo] = liWork;

	return liMNo;
}

// ヒープ - キュー追加
int
fHeapEnqueue(
	int piVal					// <I> 値
)
{
	int liRet;

	// 末尾に追加
	si1Heap[siHCnt] = piVal;
	siHCnt++;

	// 親子関係チェック
	int liNo = siHCnt - 1;
	while (1) {

		// 親の配列番号
		liNo = (liNo - 1) / 2;

		// 親子関係チェック
		liRet = fHeapChk(liNo);
		if (liRet < 0) {
			break;
		}
	}

	return 0;
}

// ヒープ - キュー取得
int
fHeapDequeue(
	int *pipRet					// <O> 取得値
)
{
	// データ数
	if (siHCnt < 1) {
		return -1;
	}

	// 取得
	*pipRet = si1Heap[0];
	siHCnt--;

	// データ数
	if (siHCnt < 1) {
		return 0;
	}

	// 末尾を先頭へ
	si1Heap[0] = si1Heap[siHCnt];

	// 親子関係チェック
	int liNo = 0;
	while (liNo >= 0) {
		liNo = fHeapChk(liNo);
	}

	return 0;
}

// 実行メイン
long long
fMain(
)
{
	int i;
	char lc1Buf[1024];

	// 品物数・割引数 - 取得
	int liICnt, liDCnt;
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);
	sscanf(lc1Buf, "%d%d", &liICnt, &liDCnt);

	// 品物 - 取得
	int liVal;
	for (i = 0; i < liICnt; i++) {
		fscanf(szpFpI, "%d", &liVal);
		fHeapEnqueue(liVal);
	}
	fgets(lc1Buf, sizeof(lc1Buf), szpFpI);

	// 割引 - 実行
	for (i = 0; i < liDCnt; i++) {
		fHeapDequeue(&liVal);
		fHeapEnqueue(liVal / 2);
	}

	// 合計値 - 取得
	long long llSum = 0;
	for (i = 0; i < liICnt; i++) {
		fHeapDequeue(&liVal);
		llSum += liVal;
	}

	return llSum;
}

// １回実行
int
fOne(
)
{
	long long llRet;
	char lc1Buf[1024];

	// データ - 初期化
	siHCnt = 0;													// ヒープ数

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
	llRet = fMain();

	// 出力
	sprintf(lc1Buf, "%lld\n", llRet);
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


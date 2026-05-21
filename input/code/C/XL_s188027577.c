#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 内部定数
#define D_CARD_MAX		50													// 最大カード枚数
#define D_CARD_VAL		50													// カード最大値
#define D_TREE_WCNT		2													// 木の方向数
#define D_TREE_LEFT		0													// 木の方向 - 左側
#define D_TREE_RIGHT	1													// 木の方向 - 右側
#define D_TREE_CNT		2													// 木の数
#define D_TNO_A			0													// 木の配列番号 - A
#define D_TNO_B			1													// 木の配列番号 - B

// 内部構造体 - 木構造
typedef struct Tree {
	int miSum;																// 合計値
	long long mlCnt;														// データ数
	int mi1Height[D_TREE_WCNT];												// 木の高さ
	struct Tree *mzp1Child[D_TREE_WCNT];									// 子
} Tree;

// 内部変数
static Tree sz2Tree[D_TREE_CNT][D_CARD_VAL * D_CARD_MAX];					// 木の実データ
static int si1TCnt[D_TREE_CNT];												// 木の実データ数
static Tree *szp1Top[D_TREE_CNT];											// 先頭の木データ

// 内部変数 - テスト用
#ifdef D_TEST
	static int siRes;
	static FILE *szpFpT, *szpFpA;
#endif

// 木データ - クリア
int
fTreeClear(
	int piTNo					// <I> 木の配列番号 0～
)
{
	si1TCnt[piTNo] = 0;			// 木の実データ数
	szp1Top[piTNo] = NULL;		// 先頭の木データ

	return 0;
}

// 木データ - 作成
Tree *
fTreeMake(
	int piTNo					// <I> 木の配列番号 0～
	, int piSum					// <I> 合計値
	, long long plCnt			// <I> データ数
)
{
	// 対象の木データ
	Tree *lzpTree = &(sz2Tree[piTNo][si1TCnt[piTNo]]);
	(si1TCnt[piTNo])++;

	// データセット
	memset(lzpTree, 0, sizeof(Tree));		// 初期化
	lzpTree->miSum = piSum;					// 合計値
	lzpTree->mlCnt = plCnt;					// データ数

	return lzpTree;
}

// 木データ - 比較
int
fTreeCmp(
	int piTNo					// <I> 木の配列番号 0～
	, int piSum					// <I> 合計値
	, Tree *pzpTree				// <I> 木データ
)
{
	// 合計値
	if (piSum < pzpTree->miSum) {
		return -1;
	}
	else if (piSum > pzpTree->miSum) {
		return 1;
	}

	return 0;
}

// 木データ - 高さ取得
int
fTreeGetHeight(
	Tree *pzpTree				// <I> 対象の木情報
)
{
	// データ有無
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

// 木データ - 右回転(親は子の右下へ・子は親の左上へ)
int
fTreeRttR(
	Tree **pzppTree				// <I> 回転対象
)
{
	// 現在の子
	Tree *lzpChild = (*pzppTree)->mzp1Child[D_TREE_LEFT];

	// 右回転
	(*pzppTree)->mzp1Child[D_TREE_LEFT] = lzpChild->mzp1Child[D_TREE_RIGHT];	// 親の左側 = 子の右側
	(*pzppTree)->mi1Height[D_TREE_LEFT] = lzpChild->mi1Height[D_TREE_RIGHT];	// 親の高さ(左) = 子の高さ(右)
	lzpChild->mzp1Child[D_TREE_RIGHT] = *pzppTree;								// 子の右側 = 親
	lzpChild->mi1Height[D_TREE_RIGHT] = fTreeGetHeight(*pzppTree);				// 子の高さ(右) - 親の高さ
	*pzppTree = lzpChild;														// 親 = 子
	
	return 0;
}

// 木データ - 左回転(親は子の左下へ・子は親の右上へ)
int
fTreeRttL(
	Tree **pzppTree				// <I> 回転対象
)
{
	// 現在の子
	Tree *lzpChild = (*pzppTree)->mzp1Child[D_TREE_RIGHT];

	// 左回転
	(*pzppTree)->mzp1Child[D_TREE_RIGHT] = lzpChild->mzp1Child[D_TREE_LEFT];	// 親の右側 = 子の左側
	(*pzppTree)->mi1Height[D_TREE_RIGHT] = lzpChild->mi1Height[D_TREE_LEFT];	// 親の高さ(右) = 子の高さ(左)
	lzpChild->mzp1Child[D_TREE_LEFT] = *pzppTree;								// 子の左側 = 親
	lzpChild->mi1Height[D_TREE_LEFT] = fTreeGetHeight(*pzppTree);				// 子の高さ(左) - 親の高さ
	*pzppTree = lzpChild;														// 親 = 子

	return 0;
}

// 木データ - 追加・削除の共通処理
// 戻り値：[1]高さの変更あり [0]高さの変更なし
int
fTreeComAddDel(
	Tree **pzppNow				// <I> 現在の木情報
	, int piWay					// <I> 対象の方向
)
{
	// 高さの変更があるかチェック
	int liNew = fTreeGetHeight((*pzppNow)->mzp1Child[piWay]);
	if ((*pzppNow)->mi1Height[piWay] == liNew) {												// 変化なし
		return 0;
	}
	(*pzppNow)->mi1Height[piWay] = liNew;														// 更新

	// 高さが離れている場合、回転
	if ((*pzppNow)->mi1Height[D_TREE_LEFT] - (*pzppNow)->mi1Height[D_TREE_RIGHT] > 1) {			// 左側が高い
		fTreeRttR(pzppNow);																			// 右回転
	}
	else if ((*pzppNow)->mi1Height[D_TREE_RIGHT] - (*pzppNow)->mi1Height[D_TREE_LEFT] > 1) {	// 右側が高い
		fTreeRttL(pzppNow);																			// 左回転
	}

	return 1;
}

// 木データ - 追加
// 戻り値：[1]高さの変更あり [0]高さの変更なし
int
fTreeAdd(
	int piTNo					// <I> 木の配列番号 0～
	, Tree **pzppNow			// <I> 現在の木情報
	, int piSum					// <I> 合計値
	, long long plCnt			// <I> データ数
)
{
	// 作成
	if (*pzppNow == NULL) {
		*pzppNow = fTreeMake(piTNo, piSum, plCnt);
		return 1;
	}

	// 比較
	int liRet = fTreeCmp(piTNo, piSum, *pzppNow);
	if (liRet == 0) {																	// 一致
		(*pzppNow)->mlCnt += plCnt;															// データ数
		return 0;
	}

	// 方向の判別
	int liWay;
	if (liRet < 0) {																	// 左側
		liWay = D_TREE_LEFT;
	}
	else {																				// 右側
		liWay = D_TREE_RIGHT;
	}

	// 下位へ
	liRet = fTreeAdd(piTNo, &((*pzppNow)->mzp1Child[liWay]), piSum, plCnt);
	if (liRet == 0) {																	// 変更なし
		return 0;
	}

	// 追加・削除の共通処理
	return fTreeComAddDel(pzppNow, liWay);
}

// 実行メイン
int
fMain(
	int piTNo					// <I> テスト番号 1～
)
{
	int i, j, k;
	char lc1Buf[1024], lc1Out[1024];

	// テストファイルオープン
#ifdef D_TEST
	sprintf(lc1Buf, ".\\Test\\T%d.txt", piTNo);
	szpFpT = fopen(lc1Buf, "r");
	sprintf(lc1Buf, ".\\Test\\A%d.txt", piTNo);
	szpFpA = fopen(lc1Buf, "r");
	siRes = 0;
#endif

	// カード枚数・平均値取得
	int liCCnt, liAvg;
#ifdef D_TEST
	fgets(lc1Buf, sizeof(lc1Buf), szpFpT);
#else
	fgets(lc1Buf, sizeof(lc1Buf), stdin);
#endif
	sscanf(lc1Buf, "%d%d", &liCCnt, &liAvg);

	// 木の初期データ
	int liFNo = D_TNO_A;
	int liTNo = D_TNO_B;
	fTreeClear(liFNo);								// クリア
	fTreeAdd(liFNo, &szp1Top[liFNo], 0, 1);			// 追加

	// カード取得
	for (i = 0; i < liCCnt; i++) {
		int liCard;
#ifdef D_TEST
		fscanf(szpFpT, "%d", &liCard);
#else
		fscanf(stdin, "%d", &liCard);
#endif
		liCard -= liAvg;

		// 木データ(先)のクリア
		fTreeClear(liTNo);

		// 木データ(元)数でループ
		for (j = 0; j < si1TCnt[liFNo]; j++) {
			Tree *lzpTree = &sz2Tree[liFNo][j];

			// 追加値(カードの選択あり・なし)
			int li1Sum[2];
			li1Sum[0] = lzpTree->miSum;
			li1Sum[1] = li1Sum[0] + liCard;

			// 追加
			for (k = 0; k < 2; k++) {
				fTreeAdd(liTNo, &szp1Top[liTNo], li1Sum[k], lzpTree->mlCnt);
			}
		}

		// 次回用
		if (liFNo == D_TNO_A) {
			liFNo = D_TNO_B;
			liTNo = D_TNO_A;
		}
		else {
			liFNo = D_TNO_A;
			liTNo = D_TNO_B;
		}
	}
#ifdef D_TEST
	fgets(lc1Buf, sizeof(lc1Buf), szpFpT);
#else
	fgets(lc1Buf, sizeof(lc1Buf), stdin);
#endif

	// 合計パターン数取得
	long long llSum = 0;
	for (i = 0; i < si1TCnt[liFNo]; i++) {
		Tree *lzpTree = &sz2Tree[liFNo][i];

		// 対象チェック
		if (lzpTree->miSum != 0) {
			continue;
		}

		// 加算
		llSum += lzpTree->mlCnt;
	}

	// 結果セット
	sprintf(lc1Out, "%lld\n", llSum - 1);

	// 結果表示
#ifdef D_TEST
	fgets(lc1Buf, sizeof(lc1Buf), szpFpA);
	if (strcmp(lc1Buf, lc1Out)) {
		siRes = -1;
	}
#else
	printf("%s", lc1Out);
#endif

	// テストファイルクローズ
#ifdef D_TEST
	fclose(szpFpT);
	fclose(szpFpA);
#endif

	// テスト結果
#ifdef D_TEST
	if (siRes == 0) {
		printf("OK %d\n", piTNo);
	}
	else {
		printf("NG %d\n", piTNo);
	}
#endif

	return 0;
}

int
main()
{

#ifdef D_TEST
	int i;
	for (i = D_TEST_SNO; i <= D_TEST_ENO; i++) {
		fMain(i);
	}
#else
	fMain(0);
#endif

	return 0;
}


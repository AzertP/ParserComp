#include <stdio.h>

int main(void) {
	/* ローカル変数 */
	int l_i_one_cards;  /* 1が書かれたカード枚数 */
	int l_i_zero_cards;  /* 0が書かれたカード枚数 */
	int l_i_minus_one_cards;  /* -1が書かれたカード枚数 */
	int l_i_get_cards;  /* 選ぶカード枚数 */
	int l_i_remaining_get_cards;  /* 残りの選ぶカード枚数 */
	int l_i_get_one_cards;  /* 1を選んだ枚数 */
	int l_i_get_zero_cards;  /* 0を選んだ枚数 */
	int l_i_get_minus_one_cards;  /* -1を選んだ枚数 */
	int l_i_max;  /* 最大値 */

	/* 各値を入力 */
	scanf("%d %d %d %d", &l_i_one_cards, &l_i_zero_cards, &l_i_minus_one_cards, &l_i_get_cards);

	/* 残りの選ぶカード枚数を取得 */
	l_i_remaining_get_cards = l_i_get_cards;

	/* 1のカードを選ぶ */
	if (l_i_remaining_get_cards < l_i_one_cards) {
		l_i_get_one_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_one_cards = l_i_one_cards;
	}
	l_i_remaining_get_cards -= l_i_get_one_cards;

	/* 0のカードを選ぶ */
	if (l_i_remaining_get_cards < l_i_zero_cards) {
		l_i_get_zero_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_zero_cards = l_i_zero_cards;
	}
	l_i_remaining_get_cards -= l_i_get_zero_cards;

	/* -1のカードを選ぶ */
	if (l_i_remaining_get_cards < l_i_minus_one_cards) {
		l_i_get_minus_one_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_minus_one_cards = l_i_minus_one_cards;
	}
	l_i_remaining_get_cards -= l_i_get_minus_one_cards;

	/* 最大値を算出 */
	l_i_max = l_i_get_one_cards - l_i_get_minus_one_cards;

	/* 表示 */
	printf("%d", l_i_max);

	return 0;
}

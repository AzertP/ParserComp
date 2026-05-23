
int main(void) {
	/*  */
	int l_i_one_cards;  /* 1 */
	int l_i_zero_cards;  /* 0 */
	int l_i_minus_one_cards;  /* -1 */
	int l_i_get_cards;  /*  */
	int l_i_remaining_get_cards;  /*  */
	int l_i_get_one_cards;  /* 1 */
	int l_i_get_zero_cards;  /* 0 */
	int l_i_get_minus_one_cards;  /* -1 */
	int l_i_max;  /*  */

	/*  */
	scanf("%d %d %d %d", &l_i_one_cards, &l_i_zero_cards, &l_i_minus_one_cards, &l_i_get_cards);

	/*  */
	l_i_remaining_get_cards = l_i_get_cards;

	/* 1 */
	if (l_i_remaining_get_cards < l_i_one_cards) {
		l_i_get_one_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_one_cards = l_i_one_cards;
	}
	l_i_remaining_get_cards -= l_i_get_one_cards;

	/* 0 */
	if (l_i_remaining_get_cards < l_i_zero_cards) {
		l_i_get_zero_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_zero_cards = l_i_zero_cards;
	}
	l_i_remaining_get_cards -= l_i_get_zero_cards;

	/* -1 */
	if (l_i_remaining_get_cards < l_i_minus_one_cards) {
		l_i_get_minus_one_cards = l_i_remaining_get_cards;
	}
	else {
		l_i_get_minus_one_cards = l_i_minus_one_cards;
	}
	l_i_remaining_get_cards -= l_i_get_minus_one_cards;

	/*  */
	l_i_max = l_i_get_one_cards - l_i_get_minus_one_cards;

	/*  */
	printf("%d", l_i_max);

	return 0;
}

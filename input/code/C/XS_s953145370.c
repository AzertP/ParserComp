#include <stdio.h>
int main() {
	char s[4];
	int ans = 0;
	scanf("%s", s);
	for (int i = 0; i < 3; i++) {
		if (s[i] == '1') ans++;
	}
	printf("%d\n", ans);
	return 0;
}
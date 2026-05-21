/*
 * main.c
 *
 *  Created on: 2019/07/21
 *      Author: family
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main()
{
	char S[101] = {0};
	int N = 0;
	scanf("%d", &N);
	scanf("%s", S);
	if (N%2 != 0) {
		printf("No\n");
	} else {
		int ret = 0;
		ret = strncmp(&S[0], &S[(N/2)], N/2);
		if (ret != 0) {
			printf("No\n");
		} else {
			printf("Yes\n");
		}
	}
    return 0;
}

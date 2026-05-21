#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h> // uint64_t

#define STR_MAX 3000
#define BUF_SIZE (STR_MAX+50)

// size: specify sizeof(str)
int get_str(char *str, int size) {
    if(!fgets(str, size, stdin)) return -1;
    return 0;
}

#define max(a,b) ((a) > (b) ? (a) : (b))

int main(void) {
    char astr[BUF_SIZE];
    char bstr[BUF_SIZE];
    static int dp[STR_MAX+1][STR_MAX+1];
    static char trace[STR_MAX+1][STR_MAX+1];
    get_str(&astr[1], BUF_SIZE-1);
    get_str(&bstr[1], BUF_SIZE-1);
    int alen = strlen(&astr[1])-1; int blen = strlen(&bstr[1])-1;
    int i, j;
#ifdef DEBUG
    printf("%d %d\n", alen, blen);
#endif

    for(i = 1; i <= alen; i++) {
        for(j = 1; j <= blen; j++) {
            if(dp[i-1][j] >= dp[i][j-1]) {
                dp[i][j] = dp[i-1][j];
                trace[i][j] = 1;
            } else { dp[i][j] = dp[i][j-1]; }
            if(astr[i] == bstr[j]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                trace[i][j] = astr[i];
            }
        }
    }
#ifdef DEBUG
    printf("%d\n", dp[alen][blen]); // length(LCS)
#endif
    int ait = alen; int bit = blen;
    char stack[BUF_SIZE];
    int sidx = 0;
    while(ait >= 1 && bit >= 1) {
#ifdef DEBUG
        printf("(%d, %d): %d\n", ait, bit, trace[ait][bit]);
#endif
        if(trace[ait][bit]>1) {
            stack[sidx++] = trace[ait][bit];
            ait--; bit--;
        } else if(trace[ait][bit] == 1) {
            ait--;
        } else { bit--; }
    }
    // presentation
    for(i = sidx-1; i >= 0; i--) {
        putchar(stack[i]);
    }
    putchar('\n');
    return 0;
}
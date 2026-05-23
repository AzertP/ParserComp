
int main(void) {
    const int merry[] = {1, 2, 1, 2, 1, 4, 1, 4};
    int p[8], i, j, k, d, dmin, check, min_check;

    while (scanf("%d", &p[0]) == 1) {
        for (i = 1; i < 8; i++) {
            scanf("%d", &p[i]);
        }
        for (i = 0, dmin = INT_MAX, k = 0, min_check = 9999;
             i < 8; i++) {
            for (j = 0, d = 0; j < 8; j++) {
                d += p[j] > merry[(i+j)%8]? p[j]-merry[(i+j)%8]: 0;
            }
            check = merry[i]*1000 + merry[(i+1)%8]*100 + merry[(i+2)%8]*10 + merry[(i+3)%8];
            if (d < dmin || (d == dmin && check < min_check)) {
                dmin = d;
                min_check = check;
                k = i;
            }
        }
        printf("%d", merry[k]);
        for (i = 1; i < 8; i++) {
            printf(" %d", merry[(k+i)%8]);
        }
        puts("");
    }
    return 0;
}

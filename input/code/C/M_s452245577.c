#include <stdio.h>
#include <stdlib.h>

int main (void) {
    int h, w;

    scanf("%d%d", &h, &w);

    char a[h][w+1];

    for ( int i=0; i<h; i++ )
        scanf("%s", a[i]);

    /*
    for ( int i=0; i<h; i++ ) {
        printf("%s\n", a[i]);
    }
     */

    int row, col;
    char skip_row[h], skip_col[w];

    /* initialize */
    for ( row=0; row<h; row++ )
        skip_row[row] = 1;
    for ( col=0; col<w; col++ )
        skip_col[col] = 1;

    /* 各行/列をスキップできるかチェック */
    for ( row=0; row<h; row++ ) {
        for ( col=0; col<w; col++ ) {
            if ( a[row][col] == '#' ) {
                skip_row[row] = 0;
                skip_col[col] = 0;
            }
        }
    }

    /* 出力 */
    for ( row=0; row<h; row++ ) {
        if ( skip_row[row] == 1 )
            continue;
        for ( col=0; col<w; col++ ) {
            if ( skip_col[col] == 1 )
                continue;
            printf("%c", a[row][col]);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}

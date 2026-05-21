#include <stdio.h>
#include <string.h>
 
#define SENTINEL 100000000
#define N 100000
 
typedef struct{
    char c;
    int num;
} Card;
 
Card card1[N];
Card card2[N];
 
void merge(int left, int mid, int right) {
    int n1 = mid - left;
    int n2 = right - mid;
    Card L[n1], R[n2];
    int i, j, k;
 
    for (i = 0; i <= n1 - 1; i++)
        L[i] = card2[left + i];
    for (i = 0; i <= n2 - 1; i++)
        R[i] = card2[mid + i];
 
    L[n1].num = R[n2].num = SENTINEL;
    L[n1].c = R[n2].c = ' ';
    i = j = 0;
 
    for (k = left; k <= right - 1; k++) {
        if (L[i].num <= R[j].num) {
            card2[k] = L[i];
            i++;
        } else {
            card2[k] = R[j];
            j++;
        }
    }
}
 
void mergeSort(int left, int right) {
    int mid;
 
    if (left + 1 < right) {
        mid = (left + right) / 2;
        mergeSort(left, mid);
        mergeSort(mid, right);
        merge(left, mid, right);
    }
 
}
 
void swapOfInt(int *a, int *b) {
    int tmp;
 
    tmp = *a;
    *a = *b;
    *b = tmp;
}
 
void swapOfChar(char *a, char *b) {
    char tmp;
 
    tmp = *a;
    *a = *b;
    *b = tmp;
}
 
int partition(int p, int r) {
    int x, i, j;
 
    x = card1[r].num;
    i = p - 1;
 
    for (j = p; j < r; ++j) {
        if (card1[j].num <= x) {
            i++;
            swapOfInt(&card1[i].num, &card1[j].num);
            swapOfChar(&card1[i].c, &card1[j].c);
        }
    }
    swapOfInt(&card1[i + 1].num, &card1[r].num);
    swapOfChar(&card1[i + 1].c, &card1[r].c);
    return i + 1;
}
 
void quickSort(int p, int r) {
    int q;
 
    if (p < r) {
        q = partition(p, r);
        quickSort(p, q - 1);
        quickSort(q + 1, r);
    }
}
 
int main() {
    int i, n, flag = 0;
    char ch;
 
    scanf("%d%c", &n, &ch);
 
    for (i = 0; i < n; ++i) {
        scanf("%c%d%c", &card1[i].c, &card1[i].num, &ch);
        card2[i] = card1[i];
    }
 
    quickSort(0, n - 1);
    mergeSort(0, n);
 
    for (i = 0; i < n; i++) {
        if (strcmp(&card1[i].c, &card2[i].c)) flag = 1;
    }
 
    if (flag) printf("Not stable\n");
    else printf("Stable\n");
 
    for (i = 0; i < n; ++i) {
        printf("%c %d\n", card1[i].c, card1[i].num);
    }
 
    return 0;
}
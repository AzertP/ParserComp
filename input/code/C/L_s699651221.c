#include<stdio.h>
//10万
#define MAX 100000
//20億
#define SENTINEL 2000000000

struct Card{
  char suit;
  int value;
};

//配列LとR、10億+2を用意
struct Card L[MAX / 2 + 2], R[MAX / 2 + 2];

void merge(struct Card A[], int n, int left, int mid, int right) {
  int i, j, k;
  int n1 = mid - left;
  int n2 = right - mid;
  for( i = 0; i < n1; i++ ) L[i] = A[left + i]; //配列Aの左側の各要素を配列Lに代入
  for( i = 0; i < n2; i++ ) R[i] = A[mid + i]; //配列Aの右側の各要素を配列Rに代入
  L[n1].value = R[n2].value = SENTINEL; //番兵
  i = j = 0;
  for( k = left; k < right; k++ ) {
    //LとRの配列を比較して、昇順になるようにAの配列に統合する
    if( L[i].value <= R[j].value ) {
      A[k] = L[i++];
    } else {
      A[k] = R[j++];
    }
  }
}
//mergeSort(A, n, 0, n);
//配列全体を対象として mergeSort を行う
void mergeSort(struct Card A[], int n, int left, int right) {
  int mid;
  //要素が2個以上の時、それぞれ n/2 個の要素を含む2つの部分配列に分割する
  if( left+1 < right ) {
    mid = (left + right) / 2;
    mergeSort(A, n, left, mid); //配列左側の分割
    mergeSort(A, n, mid, right); //配列右側の分割
    merge(A, n, left, mid, right);
  }
}

int partition(struct Card A[], int n, int p, int r) {
  int i, j;
  struct Card t, x;
  x = A[r]; //パーティションの基準となるA[r]をxに代入
  i = p - 1;
  for( j = p; j < r; j++) {
    //パーティションの基準となるxとA[j]を比較
    if( A[j].value <= x.value ) {
      //A[j]がx以下の時
      i++;
      t = A[i];
      A[i] = A[j];
      A[j] = t;
    }
  }
  //A[j]がxより大きい時
  t = A[i+1];
  A[i+1] = A[r];
  A[r] = t;
  return i+1;
}
//quickSort(B, n, 0, n-1)
void quickSort(struct Card A[], int n, int p, int r) {
  int q;
  if( p < r ) {
    q = partition(A, n, p, r);
    quickSort(A, n, p, q-1);
    quickSort(A, n, q+1, r);
  }
}

int main() {
  int n, i, v;
  struct Card A[MAX], B[MAX]; //AとBの配列を用意する
  char S[10];
  int stable = 1; //安定している時を"1"とする

  scanf("%d", &n); //カードの枚数

//カードの枚数分forを回す
  for( i = 0; i < n; i++ ) {
    scanf("%s %d", S, &v); //絵柄と数を入力
    A[i].suit = B[i].suit = S[0]; //それぞれの配列に絵柄を代入
    A[i].value = B[i].value = v; //それぞれの配列に数を代入
  }

  mergeSort(A, n, 0, n);
  quickSort(B, n, 0, n-1);

  for( i = 0; i < n; i++ ) {
    //マージソートとクイックソートの結果を比べる
    if( A[i].suit != B[i].suit ) stable = 0;
  }

  if( stable == 1 ) printf("Stable\n");
  else printf("Not stable\n");
  for( i = 0; i < n; i++ ) {
    printf("%c %d\n", B[i].suit, B[i].value); //ソートした結果を示す
  }

  return 0;
}


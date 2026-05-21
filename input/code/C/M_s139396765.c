#include <stdio.h>

#define L 1000000


int main() {

  int n,S[L],q,T[L];
  int i,num=0;
  int mid;
  int right,left;

  scanf("%d", &n);

  for( i = 0; i < n; i++) {
    scanf("%d", &S[i]);
  }

  scanf("%d", &q);

  for (i = 0; i < q; i++) {
    /* code */
    scanf("%d", &T[i]);
  }

  //BainarySearch
  for (i = 0; i < q; i++) {
    /* code */

  left = 0;
  right = n;
//  printf("A\n");
    while (left<right) {
    //  printf("B\n");
      /* code */
      mid=(left+right)/2;
      if(S[mid] == T[i]){
        left=mid;
        right=mid;
       num++;
    }else if(T[i]<S[mid]){
      right=mid;
    }else if(T[i]>S[mid]){
      left=mid+1;
    }
  }
//  printf("C\n");

}
  printf("%d\n", num);



  return 0;
}


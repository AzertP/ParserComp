#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int compare_int(const void *a, const void *b)
{
    return *(int*)a - *(int*)b;
//    return *(long long*)a - *(long long*)b;
}
//  qsort(A, b, sizeof(int), compare_int);

int main()
{
 
//  int a, b, c, d, e, X, Y;
  int a, b, c,  d, e,f;
  //  double  c;
//  long long  A[1000001]={};
  int  A[2000001]={};
  int  B[2000001]={};
  int  ans[100003]={};
  int  sum[100003]={};
  int  cnt_flag[100003]={};

  int  L[1003]={};


//  long long  a, b, c, d, X, Y;
//  long long   d, X, Y;
  long long JPY;
  double BTC;
  char s[200001]={};
  char t[200001]={};
  char u[200001]={};
//  int N, M;
//  int A[1000000001]={};
//  long long  C[101]={};
  int i,j,k;
  int flag=0;
//  int flag[10000003]={};
  int tmp=0;
//  int min=1000000;
  int min=10;
  int len=0;
  int cnt=0;
  int cnt0=0;
  int cnt1=0;
  int num=0;
//  double ans=0.0;
//  long long ans=0;
  int max=0;
//  int sum=0;
//  char alp[26]={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};

  //  a=0;
//  scanf("%lld %lld", &a, &b);
 

  scanf("%s", &s);
  len=strlen(s);
//  scanf("%d", &max);
  for(i=0;i<len;i++){
    if(s[i-1]=='0'){
//  printf("%d\n", cnt);
      if(s[i]=='0'){
        s[i]='1';
        cnt++;
      }
    }
    if(s[i-1]=='1'){
      if(s[i]=='1'){
        s[i]='0';
        cnt++;
      }
    }
  }
  
  printf("%d\n", cnt);
/*    
    */

//  printf("%s\n",s);
//    else  
  
return 0;
}

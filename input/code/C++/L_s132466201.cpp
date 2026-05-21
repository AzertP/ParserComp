#include <cstdio>
#include <cstring>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <stack>
#include <queue>
using namespace std;



int main(){
  long long int  n,k,sum;
  sum = 0;
  cin >> n >> k;
  long long int a[n];
  for(int i=0;i<n;i++){
    cin >> a[i];
    sum += a[i];
  }

  sort(a,a+n);

//枝刈
  if(n==1){
    if(k<=a[0])
      cout << 0 << endl;
    else
      cout << 1 << endl;
    return 0;
  }

  if(sum < k){
    cout << n << endl;
    return 0;
  }

  if(a[0]>=k){
    cout << 0 << endl;
    return 0;
  }


//本作業
  int need[5001]={};

  for(int i=0;i<n;i++){
    if(i>0 && a[i] == a[i-1]){
      need[i] = need[i-1];
    }else if(a[i]>=k){
      need[i] = 1;
    }
    else{
      int min;
        if(k - a[i] < 0 )
          min = 0;
        else
          min = k - a[i];
      int max = k - 1;

      bool flag = false;
      bool prob[5001]={};
      prob[0] = 1;
      int edge = 0;

      for(int j=0;j<n;j++){

        if(i!=j){
          if(edge + a[j] <= k)
            edge += a[j];
          else
            edge = k;

          for(int l= edge ;l>=0;l--){

            if((l - a[j] >=0) && (prob[l - a[j]] == 1)){

              prob[l] = 1;

              if(min <= l && l <= max){
                need[i] = 1;
                flag = true;
              }
            }

            if(flag) break;
          }
        }

        if(flag) break;
      }
    }
  }
  int ans = 0;
  for(int i=0;i<n;i++)
    if(need[i] == 0)
      ans++;

  cout << ans <<endl;

  return 0;
}

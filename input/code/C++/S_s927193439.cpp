#include <cstdio>
#include <iostream>
using namespace std;

int main(){
  int a,b,c,d,res;
  cin >> a;
  b=a%1000/100;
  c=a%100/10;
  d=a%10;
  a/=1000;
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
      for(int k=0;k<2;k++){
        for(int l=0;l<2;l++){
          res=a;
          res+=(j)?-b:b;
          res+=(k)?-c:c;
          res+=(l)?-d:d;
          if(res==7){
            printf("%d%c%d%c%d%c%d=7\n",a,(j)?'-':'+',b,(k)?'-':'+',c,(l)?'-':'+',d);
            return 0;
          }
        }
      }
    }
  }
}

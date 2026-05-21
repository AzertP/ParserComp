#include<bits/stdc++.h>
using namespace std;
int main(){
  long long p=100,m=0,x;
  cin>>x;
  while(p<x){
    p=p+((p)/100);
    m++;
  }cout<<m;
  return 0;
}
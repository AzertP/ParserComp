#include <bits/stdc++.h>
#define rep(i,n) for(int i=0;i<n;i++)
typedef long long ll;
using namespace std;

int main(){
  string s;
  cin >> s;
  int n = s.size();
  vector<int> c(n);
  
  rep(i,2){
    int cnt = 0;
    rep(j,n){
      if(s[j]=='R'){
        cnt++;
      }else{
        c[j] += cnt/2;
        c[j-1] += (cnt+1)/2;
        cnt = 0;
      }
    }
    reverse(c.begin(),c.end());
    reverse(s.begin(),s.end());
    rep(k,n){
      if(s[k]=='R')s[k]='L';
      else s[k] = 'R';
    }
  }
  rep(i,n){
    printf("%d\n",c[i]);
  }
}

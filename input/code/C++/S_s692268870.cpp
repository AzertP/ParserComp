#include<bits/stdc++.h>
#define rep(i,n) for(int i = 0; i < n; i++)
#define pb push_back
using namespace std;
typedef long long ll;

string s="zabcdefghijklmnopqrstuvwxy";

int main(){
  ll n;
  cin>>n;
  vector<int> p;
  ll b=n;;
  while(b>=26){
    p.pb(b%26);
    if(b%26==0) b=b/26-1;
    else b/=26;
  }
  if(b) p.pb(b);
  vector<char> ans;
  rep(i,p.size()){
    ans.pb(s[p[i]]);
  }
  for(int i=ans.size()-1;i>=0;i--) cout<<ans[i];
}
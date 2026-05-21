#include<bits/stdc++.h>
using namespace std;

int main(){
  string s,d,f; cin>>s>>d>>f;
  int sa = 'A'-'a';
  cout<<(char)(s.at(0)+sa)<<(char)(d.at(0)+sa)<<(char)(f.at(0)+sa)<<endl;
}
#include <bits/stdc++.h>
#define ll long long
#define pi 3.14159265358979
#define mod 1000000007
#define rep(i,n) for(ll i=0;i<n;i++)
using namespace std;

int main(){
    ll i,n,a,b,k,max=0;
    cin>>a>>b>>n;
    if(b-1<=n){
      i=b-1;
    }else{
      i=n;
    }
        k=(a*i/b)-a*(i/b);
        max=k;
    cout<<max;
}
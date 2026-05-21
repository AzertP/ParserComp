#include<bits/stdc++.h>
using namespace std;
typedef long long int ll;


int main(){

    ll n,b=1e18;
    cin>>n;
    ll a[n+5];
    bool flag=false;
    for(int i=0; i<n; i++){
        cin>>a[i];
        if(a[i]==0){
            flag=true;
        }
    }
    if(flag){
        cout<<0<<endl;
        return 0;
    }

    ll m=1;
    for(int i=0; i<n; i++){
        if(m>(b/a[i])){
            flag=true;
            break;
        }
        else
            m*=a[i];
    }
    if(flag)
        cout<<-1<<endl;
    else
        cout<<m<<endl;


}

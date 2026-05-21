#include <bits/stdc++.h>
using namespace std;
#define endl "\n"
#define MOD 1000000007
#define fo(i,s,e) for( i=s;i<e;i++)
#define rfo(i,s,e) for(i=s;i>e;i--)
#define LLI long long int
#define LI long int 
#define pb push_back
#define pob pop_back()
#define sp " "
#define ff first
#define ss second
//               When something is important enough, you do it even if the odds are not in your favor.

int main(){   
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);cout.tie(NULL);
    /*#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    #endif*/
    // code goes here
    int t=1;//cin>>t;
    while(t--){
      LI n,k,i,j;cin>>n>>k;
      LI a[n];
      fo(i,0,n) cin>>a[i];
      bool dp[k+4];    // it is 0 when first wins, 1 when second wins
      dp[0]=1;     //  second player will win
      fo(i,1,k+1){  // number of stones
        dp[i]=1;
        fo(j,0,n){
          if(i-a[j]>=0&&dp[i-a[j]]) dp[i]=0;
        }
      }
      if(dp[k]) cout<<"Second";
      else cout<<"First";
    }
    return 0;
}
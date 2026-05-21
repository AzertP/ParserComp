#include<bits/stdc++.h>
using namespace std;


int main(){
	long long n;
	cin>>n;
	long long v[n];
	long long wt[n];
	long long W;
	cin>>W;
	for(long long i=0;i<n;i++){
		cin>>wt[i];

		cin>>v[i];
	}

	long long dp[n+1][W+1]={0};


	for(long long i=0;i<=n;i++){
		for(long long j=0;j<=W;j++){
			if(i==0||j==0)
				dp[i][j]=0;
			else if(wt[i-1]>j)
				dp[i][j]=dp[i-1][j];
			else{
				dp[i][j]=max(dp[i-1][j],v[i-1]+dp[i-1][j-wt[i-1]]);
			}
		}
	}

	/*for(long long i=0;i<n;i++){
		for(long long j=0;j<W;j++){
			cout<<dp[i][j]<<" ";
		}
		cout<<endl;
	}*/

	cout<<dp[n][W]<<endl;
	return 0;
}
#include <bits/stdc++.h>
using namespace std;
int main()
{
	int N,T;
	cin>>N>>T;
	if(N%T==0||T%N==0) cout<<N+T<<endl;
	else cout<<T-N<<endl;
	return 0;
}
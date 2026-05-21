#include<iostream>
using namespace std;
using ll=long long;

int n,ans=0;

int dfs(ll x,bool f7,bool f5,bool f3){
	if(x>n)return 0;
	if(f7&&f5&&f3)ans++;
	dfs(x*10+7,true,f5,f3);
	dfs(x*10+5,f7,true,f3);
	dfs(x*10+3,f7,f5,true);
	return ans;
}

int main(){
	cin>>n;
	cout<<dfs(0,0,0,0)<<endl;
}
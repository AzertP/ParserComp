#include<stack>
#include<algorithm>
#include<iostream>
using namespace std;

int main(){
	int a[10];
	stack<int> h;
	for(int i=0;i<10;++i){
		cin>>a[i];
	}
	sort(a,a+10);
	for(int i=0;i<10;++i){
		h.push(a[i]);
	}
	for(int i=0;i<3;++i){
		int ans=h.top();
		cout<<ans<<endl;
		h.pop();
	}
	return 0;
}
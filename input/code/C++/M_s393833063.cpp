#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
	while(true){
		
vector<int> order;
vector<int> store;
	int r=0;
	int d,n,m;
	cin >> d;
	if(d==0)
	break;
	cin >> n >> m;
	store.push_back(0);
	for(int i=1;i<n;i++){
	cin >> r;
	store.push_back(r);
	}
	store.push_back(d);
	for(int i=0;i<m;i++){
	cin >> r;
	order.push_back(r);
	}
	sort(store.begin(),store.end());
	sort(order.begin(),order.end());
	int now=0,sum=0;
	for(int i=0;i<m;i++){
		int p=order[i];
		while(!( p >= store[now] && p <= store[now+1]))
		now++;
		sum+=min(p-store[now],store[now+1]-p);
	}
	
	cout << sum << endl;
	}
	return 0;
}
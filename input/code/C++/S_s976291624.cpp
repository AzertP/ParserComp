#include<iostream>
#include<string>
#include<vector>
#include<queue>
#include<stack>
#include<utility>
#include<algorithm>
#include<map>
#include<set>

using namespace std;

typedef pair<int, int> P;
typedef long long int ll;

int r, D, x;
int ans[10];

int main() {
	cin >> r >> D >> x;
	ans[0] = r*x - D;
	for (int i = 1; i < 10; i++) {
		ans[i] = r*ans[i - 1] - D;
	}
	for (int i = 0; i < 10; i++) {
		cout << ans[i] << endl;
	}

}
#include <bits/stdc++.h>
using namespace std;


typedef long long ll;
typedef pair<ll,int> P;

struct edge {int to; ll cost; };

int dx[] = {-1, 1, 0,  0};
int dy[] = { 0, 0, 1, -1};
 
int main(){
  int n;
  cin >> n;
  double ans = 0.0;
  for(int i=0; i<n; i++){
    double a;
    cin >> a;
    ans += 1.0/a;
  }
  ans = 1.0/ans;
  cout << fixed << setprecision(10) << ans << endl;
  return 0;
}

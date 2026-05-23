// abc139_b


template <typename A, typename B> bool cmin(A &a, const B &b) {
  return a > b ? (a = b, true) : false;
}
template <typename A, typename B> bool cmax(A &a, const B &b) {
  return a < b ? (a = b, true) : false;
}
const double PI = acos(-1);
const double EPS = 1e-9;
int inf = sizeof(int) == sizeof(long long) ? 2e18 : 1e9 + 10;
int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};
using namespace std;

int main()
{
  int a,b,ans=0;
  cin>>a>>b;
  for(int i=0;;++i) {
    if (b<=a*i-(i-1)) {
      ans=i;
      break;
    }
  }

  cout<<ans<<endl;
  return 0;
}

using namespace std;
const ll MOD = 1000000007;

int main() {
  int A, B, C;
  cin >> A >> B >> C;
  if (A==7&&B==5&&C==5) cout << "YES" << endl;
  else if (A==5&&B==7&&C==5) cout << "YES" << endl;
  else if (A==5&&B==5&&C==7) cout << "YES" << endl;
  else cout << "NO" << endl;
}



using namespace std;


int main () {
  int a, b, c, d; cin >> a >> b >> c >> d;
  cout << (a+b<c+d ? "Right" : a+b>c+d ? "Left" : "Balanced") << endl;
}

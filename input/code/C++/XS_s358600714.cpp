using namespace std;

int main() {
  int X, A, B;
  cin >> X >> A >> B;

  X -= A;
  X %= B;
  cout << X << endl;

  return 0;
}

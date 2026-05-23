using namespace std;

main () {
  int N,A,B;
  cin >> N >> A >> B;
  if ( N * A < B ) {
    cout << N * A << endl;
  } else {
    cout << B;
  }
}

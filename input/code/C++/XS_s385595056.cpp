using namespace std;

int main() {
  int A[3]; cin >> A[0] >> A[1] >> A[2];
  sort(A,A+3);
  int answer = (A[1]-A[0]) + (A[2]-A[1]);
  cout << answer << endl;
  return 0;
}

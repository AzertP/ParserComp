using namespace std;

int main() {
  string S,T;
  cin >> S >> T;
  int count = 0;
  
  for (int i = 0; i < 3; i++) {
    if (S.at(i) == T.at(i)){
      count++;
    }
  }
  cout << count << endl;
}


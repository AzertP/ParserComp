using namespace std;



int main() {
  string s;
  cin >> s;
  int number = s.size();
  number -= 2;

  
  cout << s[0] << number << s[s.size() - 1] << endl;
}

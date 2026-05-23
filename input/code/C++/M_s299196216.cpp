typedef long long int ll;
typedef long double ld;
using namespace std;

int main(int argc, char const *argv[]) {
  ll N;
  std::cin >> N;
  string S;
  std::cin >> S;
  string T = "";
  char now = '$';
  for(int i=0;i<N;i++){
    if(now==S[i]) continue;
    else{
      T += S[i];
      now = S[i];
    }
  }
  std::cout << T.size() << '\n';
  return 0;
}

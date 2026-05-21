#include <iostream>

using namespace std;

int main(void){

  int n;
  cin >> n;
  long long ans = 1;
  for( int i = n ; i > 0 ; i-- ){
    ans *= i;
  }
  cout << ans << endl;
  return 0;
}
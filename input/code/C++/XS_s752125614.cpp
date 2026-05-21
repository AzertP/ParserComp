#include<iostream>
using namespace std;
 
int main(){
  int H, N;
  cin >> H >> N;
  int wa=0;
  int a;
  for(int i=0; i<N; i++) {
    cin >> a;
    wa += a;
  }
  if(H<=wa) cout << "Yes" << endl;
  else cout << "No" << endl;
  return 0;
}
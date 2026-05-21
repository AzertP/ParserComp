#include <bits/stdc++.h>
using namespace std;


int main() {
  int N;
  long long int Ans=0;
	cin >> N;
    vector<long long int> vec(N);
  
  	for (int i = 0; i < N; i++) {
        cin >> vec.at(i);
    }
  
    sort(vec.begin(), vec.end());
    reverse(vec.begin(), vec.end());
  
  for (int i = 1; i < N; i++) {
        Ans = Ans + vec.at(i/2);
    }
    
    cout << Ans;

}
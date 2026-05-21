#include <bits/stdc++.h>
using namespace std;
 
int main() {
    int a, b;
    cin >> a >> b;
    if(a >= b) {
        for(int i = 0; i < a; i++) {
            cout << b;
        }
    }
    else {
        for(int j = 0; j < b; j++) {
            cout << a;
        }
    }
    cout << endl;
}
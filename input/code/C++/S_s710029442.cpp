#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <unordered_map>

#define BIG 1000000007

using namespace std;

int main() {
    int a,b;
    cin >> a >> b;
    int o = max(a * 2 - 1,max(a + b,b * 2 - 1));
    cout << o;
}

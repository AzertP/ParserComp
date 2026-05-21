#include <bits/stdc++.h>
#define H ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
#define ll long long
#define ld long double
using namespace std;
const ll  mod = 998244353;

    int frq[200001];

int main()
{
    H
    int h, w, n;
    cin >> h >> w >> n;
    int x, i = 0, j = 0, arr[h][w];
    for (int q = 0; q < n; q++){
        cin >> x;
        while (x--){
           arr[i][j] = q + 1;
           i % 2 == 0 ? j++ : j--;
           if (j == w){
               i++;
               j--;
           }
           else if (j == -1){
               i++;
               j++;
           }
        }
    }
    for (int k = 0; k < h; k++){
        for (int l = 0; l < w; l++){
            cout << arr[k][l];
            if (l < w - 1) cout << ' ';
        }
        cout << '\n';
    }
    return 0 ;
}

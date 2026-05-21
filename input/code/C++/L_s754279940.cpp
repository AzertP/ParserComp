#include <bits/stdc++.h>
#define FOR(i, begin, end) for(int i=(begin);i<(end);i++)
#define REP(i, n) FOR(i,0,n)
#define int long long
using namespace std;

typedef pair<int, int> Pii;

void readint(int N, vector<int> &a);
void readdouble(int N, vector<double> &a);
void readindex(int N, vector<int> &a);



signed main(){

    int H, W, D;
    cin >> H >> W >> D;
    vector<vector<int>> A(H, vector<int>(W));
    REP(i, H){
        readindex(W, A[i]);
    }
    int Q;
    cin >> Q;
    vector<int> L(Q), R(Q);
    REP(i, Q){
        cin >> L[i] >> R[i];
        L[i]--;
        R[i]--;
    }

    vector<Pii> coor(H * W);
    REP(i, H){
        REP(j, W){
            coor[A[i][j]] = Pii(i, j);
        }
    }

    vector<int> cons(H * W, 0);

    FOR(i, D, H * W){
        int tmp = llabs(coor[i].first - coor[i - D].first) + llabs(coor[i].second - coor[i - D].second);
        cons[i] = cons[i - D] + tmp; 
    }

    REP(i, Q){
        int ans = cons[R[i]] - cons[L[i]];
        cout << ans;
        if(i != Q - 1) cout << endl;
    }
    
    return 0;
}








void readint(int N, vector<int> &a){
    string s;
    for(int i = 0; i < N - 1; i++){
        getline(cin, s, ' ');
        a[i] = atoi(s.c_str());
    }
    getline(cin, s, '\n');
    a[N - 1] = atoi(s.c_str());
}
void readdouble(int N, vector<double> &a){
    string s;
    for(int i = 0; i < N - 1; i++){
        getline(cin, s, ' ');
        a[i] = atof(s.c_str());
    }
    getline(cin, s, '\n');
    a[N - 1] = atof(s.c_str());
}
void readindex(int N, vector<int> &a){
    REP(i, N){
        cin >> a[i];
        a[i]--; 
    }
}
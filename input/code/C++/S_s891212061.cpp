#include <bits/stdc++.h>
using namespace std;
#define REP(i, n) for(int i = 0;i < n;i++)
#define REPR(i, n) for(int i = n;i >= 0;i--)
#define FOR(i, m, n) for(int i = m;i < n;i++)
typedef long long unsigned int llong;
int a_max = 1e5;

int main(){
    int n;
    cin >> n ;
    int tmp;
    vector<int> A(a_max + 1,0);
    REP(i,n){
        scanf("%d",&tmp);
        if(tmp!=0) A[tmp-1]+=1;
        A[tmp]+=1;
        if(tmp!=a_max) A[tmp+1]+=1;
    }

    auto itr = max_element(A.begin(),A.end());
    cout << *itr << endl;
    
}
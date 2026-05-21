#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

using namespace std;

typedef long long int64;
constexpr int p = 1e9+7;
constexpr long long M = 0x777777777777777777777777777777777777777LL;

int N;
int a[24];
int dp[1024*1024*4];

vector<int> V;
void combi(int cnt, int taboo=-1, int it=0, int v=0) {
    if(cnt==0) {
        V.emplace_back(v);
    }
    else for(int i=it; i<N; ++i) if(i!=taboo) {
        combi(cnt-1, taboo, i+1, v | (1<<i));
    }
}

int main() {
	scanf("%d", &N);
    dp[0] = 1;
    for(int n=0; n<N; ++n) {
        for(int n2=0; n2<N; ++n2) {
            int a;
            scanf("%d", &a);
            if(a) {
                V.clear();
                combi(n, n2);
                for(int v : V) {
                    int i = (v|(1<<n2));
                    int sum = dp[i] + dp[v];
                    if(p<=sum) sum-=p;
                    dp[i] = sum;
                }
            }
        }
    }
    printf("%lld\n", dp[(1<<N)-1]);
	return 0;
}

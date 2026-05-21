#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <vector>
#include <cstring>
#include <queue>
#include <string>
#include <algorithm>
#include <set>
using namespace std;
#define ll long long
const int maxn = 1e5 + 10;
int b[maxn] = {0}, l[maxn] = {0}, u[maxn] = {0};
long long pre[maxn] = {0}, sum[maxn] = {0};

/*long long mult(int who, int k){
    if(k >= B[who]){return U[who];}
    return L[who];
}*/

//vector <pair <long long, int> > G;

pair <long long, int> a[maxn];

/*bool check(int n, int m, int k){
    for(int i = 1; i <= m; i++){
        int p = G[i].second;
        //int p = G[i - 1].second;
        int multi = mult(p, k);
        if(sum[m + 1] - sum[i] + sum[i - 1] +
           (long long)(k - B[p]) * multi - pre[n] + pre[m + 1] >= 0){
            return true;
        }
    }
    for(int i = m + 1; i <= n; i++){
        int p = G[i].second;
        //int p = G[i - 1].second;
        int multi = mult(p, k);
        //long long a1 = sum[m] + (long long)(k - B[p]) * multi;
        //long long a2 = pre[n] - pre[i] + pre[i - 1] - pre[m];
        if(sum[m] + (long long)(k - B[p]) * multi
           - pre[n] + pre[i] - pre[i - 1] + pre[m] >= 0){return true;}
    }
    //system("pause");
    return false;
}*/

int check(int n, int m, int k)
{
	for(int i = 1; i <= m; i++)
	{
		int p = a[i].second;
		if(sum[m+1]-sum[i]+sum[i-1]+(long long)(k-b[p])*(k >= b[p] ? u[p] : l[p])-pre[n]+pre[m+1]>=0)
		{
			return true;
		}
	}
	for(int i = m+1; i <= n; i++)
	{
		int p = a[i].second;
		if(sum[m]+(long long)(k-b[p])*(k >= b[p] ? u[p] : l[p])-pre[n]+pre[i]-pre[i-1]+pre[m]>=0)
		{
			return true;
		}
	}
	return false;
}

/*int main(){
    int n, x;
    scanf("%d%d", &n, &x);
    for(int i = 1; i <= n; i++){
        scanf("%d%d%d", &B[i], &L[i], &U[i]);
        long long anw = (long long)B[i] * L[i] + (long long)((x - B[i]) * U[i]);
        //G.push_back(make_pair(anw, i));
        G[i] = make_pair(anw, i);
    }
    sort(G + 1, G + 1 + n);
    reverse(G + 1, G + 1 + n);
    //sort(G.begin(), G.end());
    //reverse(G.begin(), G.end());
    for(int i = 1; i <= n; i++){
        int j = G[i].second;
        //int j = G[i - 1].second;
        pre[i] = pre[i - 1] + (long long)(B[j] * L[j]);
        sum[i] = sum[i - 1] + (long long)((x - B[j]) * U[j]);
    }
    long long L1 = 0, R = (long long)n * x;
    while(L1 < R){
        long long mid = (L1 + R) >> 1;
        if(check(n, mid / x, mid % x)){
            R = mid;
        }
        else{
            L1 = mid + 1;
        }
    }
    printf("%lld", L1);
    return 0;
}*/

int main()
{
	int n, x;
	scanf("%d%d", &n, &x);
	for(int i = 1; i <= n; i++)
	{
		scanf("%d%d%d", &b[i], &l[i], &u[i]);
		a[i] = make_pair((ll)(x-b[i])*u[i]+(ll)b[i]*l[i], i);
	}
	sort(a+1, a+1+n);
	reverse(a+1, a+1+n);
	for(int i = 1; i <= n; i++)
	{
		pre[i] = pre[i-1]+(ll)b[a[i].second]*l[a[i].second];
		sum[i] = sum[i-1]+(ll)(x-b[a[i].second])*u[a[i].second];
	}
	ll l = 0, r = (ll)n*x;
	while(l<r)
	{
		ll mid = (l+r)>>1;
		//printf("%lld %d\n", mid, check(n, mid/x, mid%x));
		if(check(n, mid/x, mid%x))
		{
			r = mid;
		}
		else
		{
			l = mid+1;
		}
	}
	printf("%lld\n", l);
	return 0;
}

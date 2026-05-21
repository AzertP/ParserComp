/// Bismillahir Rahmanir Rahim

#include<bits/stdc++.h>
#define ll long long
#define pb push_back
#define mp make_pair
#define f0(n) for(int i=0;i<n;i++)
#define ms(x) memset(x,0,sizeof(x))
#define ins insert
#define ALL(v) v.begin(),v.end()
#define highest(x) numeric_limits<x>::max()
#define lowest(x) numeric_limits<x>::min()
#define Inf INFINITY
#define minv(v) *min_element(v.begin(),v.end())
#define maxv(v) *max_element(v.begin(),v.end())
#define PI acos(-1)
#define IOS ios::sync_with_stdio(false);
using namespace std;
int dx8[] = {0, 0, 1, 1, 1, -1, -1, -1};
int dy8[] = {1,-1, 1, -1, 0, 0, -1, 1};
int dx4[] = {0, 0, 1, -1};
int dy4[] = {1, -1, 0, 0};
const long long MOD = 1000000007;
template<typename T>inline T Bigmod(T base, T power, T MOD){
    T ret=1;
    while(power)
    {
        if(power & 1)ret=(ret*base)%MOD;
        base=(base*base)%MOD;
        power>>=1;
    }
    return ret;
}

bool sortinrev(const pair<int,int> &a,
               const pair<int,int> &b)
{
       return (a.first > b.first);
}
int N;
const int maxn=2002;
int arr[maxn];
int a[maxn][maxn];
int getGreatestSum()
{
    int prev_max = 0;
    for (int j = 0; j < N; j++)
        if (prev_max < a[N - 1][j])
            prev_max = a[N - 1][j];

    // Comparing it with the elements of above rows
    cout << prev_max << endl;
    int sum = prev_max;
    for (int i = N - 2; i >= 0; i--) {

        // Maximum of current row.
        cout << prev_max << endl;
        int curr_max = INT_MIN;
        for (int j = 0; j < N; j++)
            if (prev_max > a[i][j] && a[i][j] > curr_max)
                curr_max = a[i][j];

        // If we could not an element smaller
        // than prev_max.
        if (curr_max == INT_MIN)
            return -1;

        prev_max = curr_max;
        sum += prev_max;
    }
    return sum;
}
int main() {
    int N;
    cin >> N;
    vector<pair<ll, int>> A(N);
    for (int i = 0; i < N; i++) {
        int temp;
        cin >> temp;
        A[i] = {temp, i + 1};
    }
    sort(A.rbegin(), A.rend());
   // for(int i=0;i<N;i++)cout << A[i].first << " "<< A[i].second << endl;
    vector<vector<ll>> dp(N + 1, vector<ll>(N + 1, 0));
    for (int i = 0; i < N; i++) {
        for (int right = 0; right <= i; right++) {
            int left = i - right;
            ///cout << i << "  <<i::right>>  "<< right << " left>>  "<< left<< " :: a[i].second:: "<< A[i].second <<endl;
            dp[left + 1][right] = max(dp[left + 1][right], dp[left][right] + A[i].first * abs((left + 1) - A[i].second));
           /// cout << A[i].first*abs((left+1)-A[i].second )<< "    "<< A[i].first * abs((N-right) - A[i].second)<< endl;
            dp[left][right + 1] = max(dp[left][right + 1], dp[left][right] + A[i].first * abs((N - right) - A[i].second));
          /*  for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    cout << dp[i][j] << " ";
                }
                cout << endl;
            }
            cout << endl << endl;*/
        }
    }
    ll ans = 0;

    for (int i = 0; i <= N; i++) ans = max(ans, dp[i][N - i]);
    cout << ans << endl;
}
///Alhamdulillah

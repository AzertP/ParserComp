
using namespace std;

typedef long long ll;
typedef long double ld;
typedef unsigned long long ull;
typedef pair<int, int> pii;


//#pragma GCC optimize("Ofast")

ll binpow(ll a, ll b, ll mod = BIG_INF)
{
    ll res = 1;
    while(b)
    {
        if(b & 1)
            res *= a;
        a *= a;
        b >>= 1;
        a %= mod;
        res %= mod;
    }
    res %= mod;
    return res;
}




int T = 1, n, a, b, c, d;
string s;

bool used[SIZE];




void solve()
{
    cin >> a >> b >> c >> d;


    int lst = 0;

    while(a > 0 && c > 0)
    {
        if(!lst)
        {
            c -= b;
        }else
        {
            a -= d;
        }
        lst ^= 1;
    }

    if(a <= 0)
    {
        cout << "No\n";
    }else cout << "Yes\n";
}

int main()
{

    ios_base::sync_with_stdio(false);
    cin.tie(0);cout.tie(0);
//    freopen("input.txt", "r", stdin);
//    freopen("output.txt", "w", stdout);

//    cin >> T;

    while(T--)
    {
        solve();
    }




    return 0;
}

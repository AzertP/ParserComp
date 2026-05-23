 
using namespace std;
typedef long long int ll;
typedef long double ld;

 
ll gcd(ll a, ll b) { return b ? gcd(b,a%b) : a;}
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
typedef pair <ll,ll> P;

int main()
{
    ll N, A, B;
    cin >> N >> A >> B;
    if(abs(A-B) % 2 == 0) {
        print("Alice");
    } else {
        print("Borys");
    }
    return 0;
}

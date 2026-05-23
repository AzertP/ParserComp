using namespace std;
typedef pair<int,int> P;
typedef pair<int,P> P1;
typedef pair<P,P> P2;
typedef long long ll;

int main()
{
	while(1)
	{
		int n;
		ll res = 0;
		ll sum = 0;
		vector<ll>con;
		
		cin >> n;
		if(n == 0) break;
		for(int i=0;i<n;i++)
		{
			ll a; cin >> a;
			sum+=a;
		}
		for(int i=0;i<n-1;i++)
		{
			ll b; cin >> b;
			con.pb(b); sum+=b;
		}
		sort(con.begin(),con.end());
		res = sum;
		
		for(int i=0;i<con.size();i++)
		{
			sum -= con[i];
			res = max(res,sum*1LL*(i+2));
		}
		
		cout << res << endl;
	}
}

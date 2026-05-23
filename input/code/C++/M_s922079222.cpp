using namespace std;



int main()
{
	FASTIO;

	    ll n,m;
	     cin>>n>>m;
	     ll ans=0;

	     if(n>=1)
            ans=(n*(n-1))/2;
	     if(m>=1)
            ans+=(m*(m-1))/2;

	     cout<<ans<<"\n";


	return 0;
}




using namespace std;

int main(){
    ll i,n,a,b,k,max=0;
    cin>>a>>b>>n;
    if(b-1<=n){
      i=b-1;
    }else{
      i=n;
    }
        k=(a*i/b)-a*(i/b);
        max=k;
    cout<<max;
}

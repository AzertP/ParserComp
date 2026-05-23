using namespace std;

/*
sort(sp.begin(), sp.end(), [](PAIR l, PAIR r){
        return l.first<r.first || (l.first==r.first && l.second > r.second);
        });
*/

typedef long long  ll;


int main() {
    int k,x;
    cin>>k>>x;
    for(int i=x-k+1;i<=x+k-1;i++){
        cout<<i<<endl;
    }





    return 0;
}

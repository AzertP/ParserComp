using namespace std;

typedef long long ll;
const int maxn = 1e6+10;
const int mod = 1e9+7;

int a[maxn];

int n;

int prime[maxn/10];
bool vis[maxn];
int cnt = 0;

bool book[maxn];
bool flag  = 0;
void Init(){
	for(int i = 2; i <= maxn ; i++){
		if(vis[i]==0){
			prime[cnt++] = i;
			for(int j = i+i; j <= maxn; j += i){
				vis[j] = 1;
			}
		}
	}
}
void getFactor(int x){
	for(int i = 0; i < cnt&&prime[i]*prime[i]<=x; i++){
		if(x%prime[i]==0){
			if(book[prime[i]]==0) book[prime[i]] = 1;
			else{
				flag = 1;
				return ;
			}
			while(x%prime[i]==0) x /= prime[i];
		}
	}
	if(x>1){
		if(book[x]==0) book[x] = 1;
		else{
			flag = 1;
			return ;
		}
	}
}
int gcd(int a,int b){
	if(b==0) return a;
	return gcd(b,a%b);
}
int main(void){
	Init();
	int n;
	cin >> n;
	int ans = 0;
	for(int i = 1; i <= n; i++){
		cin >> a[i];
		ans = gcd(ans,a[i]);
		getFactor(a[i]); 
	}
	if(flag==0){
		cout<<"pairwise coprime"<<endl;
	}
	else{
		if(ans>=2){
			cout<<"not coprime"<<endl;
		}else cout<<"setwise coprime"<<endl;
	}
	return 0;
}


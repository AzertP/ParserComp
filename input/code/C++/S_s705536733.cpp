using namespace std;
typedef long long ll;
const int maxn=1000010,mod=1e9+7;
template <typename Tp> inline int getmin(Tp &x,Tp y){return y<x?x=y,1:0;}
template <typename Tp> inline int getmax(Tp &x,Tp y){return y>x?x=y,1:0;}
template <typename Tp> inline void read(Tp &x)
{
	x=0;int f=0;char ch=getchar();
	while(ch!='-'&&(ch<'0'||ch>'9')) ch=getchar();
	if(ch=='-') f=1,ch=getchar();
	while(ch>='0'&&ch<='9') x=x*10+ch-'0',ch=getchar();
	if(f) x=-x;
}
int n,sum,f[maxn];
inline int pls(int x,int y){return x+y>=mod?x+y-mod:x+y;}
int main()
{
	read(n);
	f[1]=n;f[2]=(ll)n*n%mod;
	sum=(ll)(n-1)*(n-1)%mod;
	for(rg int i=3;i<=n;i++)
	{
		sum=pls(sum,f[i-3]);
		f[i]=pls(f[i-1],pls(sum,n-i+2));
	}
	printf("%d\n",f[n]);
	return 0;
}

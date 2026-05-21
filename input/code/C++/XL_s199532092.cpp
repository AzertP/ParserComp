#include <bits/stdc++.h>
using namespace std;

using ll=int64_t;
#define int ll

#define FOR(i,a,b) for(int i=int(a);i<int(b);i++)
#define REP(i,b) FOR(i,0,b)
#define MP make_pair
#define PB push_back
#define ALL(x) x.begin(),x.end()
#ifdef MAROON_LOCAL
#define cerr (cerr<<"-- line "<<__LINE__<<" -- ")
#else
class CerrDummy{}cerrDummy;
template<class T>
CerrDummy& operator<<(CerrDummy&cd,const T&){
	return cd;
}
using charTDummy=char;
using traitsDummy=char_traits<charTDummy>;
CerrDummy& operator<<(CerrDummy&cd,basic_ostream<charTDummy,traitsDummy>&(basic_ostream<charTDummy,traitsDummy>&)){
	return cd;
}
#define cerr cerrDummy
#endif
#define REACH cerr<<"reached"<<endl
#define DMP(x) cerr<<#x<<":"<<x<<endl
#define ZERO(x) memset(x,0,sizeof(x))
#define ONE(x) memset(x,-1,sizeof(x))

using pi=pair<int,int>;
using vi=vector<int>;
using ld=long double;

template<class T,class U>
ostream& operator<<(ostream& os,const pair<T,U>& p){
	os<<"("<<p.first<<","<<p.second<<")";
	return os;
}

template<class T>
ostream& operator <<(ostream& os,const vector<T>& v){
	os<<"{";
	REP(i,(int)v.size()){
		if(i)os<<",";
		os<<v[i];
	}
	os<<"}";
	return os;
}

ll read(){
	ll i;
	scanf("%"  SCNd64,&i);
	return i;
}

void printSpace(){
	printf(" ");
}

void printEoln(){
	printf("\n");
}

void print(ll x,int suc=1){
	printf("%" PRId64,x);
	if(suc==1)
		printEoln();
	if(suc==2)
		printSpace();
}

string readString(){
	static char buf[3341000];
	scanf("%s",buf);
	return string(buf);
}

char* readCharArray(){
	static char buf[3341000];
	static int bufUsed=0;
	char* ret=buf+bufUsed;
	scanf("%s",ret);
	bufUsed+=strlen(ret)+1;
	return ret;
}

template<class T,class U>
void chmax(T& a,U b){
	if(a<b)
		a=b;
}

template<class T,class U>
void chmin(T& a,U b){
	if(b<a)
		a=b;
}

template<class T>
T Sq(const T& t){
	return t*t;
}

#define CAPITAL
void Yes(bool ex=true){
	#ifdef CAPITAL
	cout<<"YES"<<endl;
	#else
	cout<<"Yes"<<endl;
	#endif
	if(ex)exit(0);
}
void No(bool ex=true){
	#ifdef CAPITAL
	cout<<"NO"<<endl;
	#else
	cout<<"No"<<endl;
	#endif
	if(ex)exit(0);
}

const ll infLL=LLONG_MAX/3;

#ifdef int
const int inf=infLL;
#else
const int inf=INT_MAX/2-100;
#endif

bool dbg=false;

void Validate(vi x,vi y,vi d,vector<string> ans){
	assert(x.size()==ans.size());
	REP(j,x.size()){
		int u=0,v=0;
		assert(ans[j].size()==d.size());
		REP(i,d.size()){
			assert(1<=d[i]&&d[i]<=1000000000000LL);
			if(ans[j][i]=='L')
				u-=d[i];
			else if(ans[j][i]=='R')
				u+=d[i];
			else if(ans[j][i]=='D')
				v-=d[i];
			else if(ans[j][i]=='U')
				v+=d[i];
			else
				assert(false);
		}
		assert(u==x[j]&&v==y[j]);
	}
}

signed main(){
	int n=read();
	if(n<0){
		dbg=true;
		n=-n;
	}
	vi x(n),y(n);
	REP(i,n){
		if(!dbg){
			x[i]=read();
			y[i]=read();
		}else{
			x[i]=rand()%2000000000-1000000000;
			y[i]=rand()%2000000000-1000000000;
		}
		if(!dbg){
			if((x[i]+y[i]+x[0]+y[0])%2){
				cout<<-1<<endl;
				return 0;
			}
		}else{
			if((x[i]+y[i]+x[0]+y[0])%2)
				x[i]++;
		}
	}
	vi rawX=x,rawY=y;
	vi d;
	vector<string> ans(n);
	if((x[0]+y[0])%2){
		d.PB(1);
		REP(i,n){
			if(x[i]%2){
				ans[i].PB('L');
				x[i]++;
			}else{
				ans[i].PB('D');
				y[i]++;
			}
		}
	}
	cerr<<x<<endl;
	cerr<<y<<endl;
	vi u(n),v(n);
	REP(i,n){
		u[i]=x[i]+y[i];
		v[i]=-x[i]+y[i];
	}
	int len=int(1)<<35;
	//int len=4;
	while(1){
		bool ok=true;
		REP(i,n)
			ok&=(u[i]==0&&v[i]==0);
		if(ok)break;
		d.PB(len);
		REP(i,n){
			if(u[i]<=0){
				u[i]+=len;
				if(v[i]<=0){
					v[i]+=len;
					ans[i].PB('D');
				}else{
					v[i]-=len;
					ans[i].PB('L');
				}
			}else{
				u[i]-=len;
				if(v[i]<=0){
					v[i]+=len;
					ans[i].PB('R');
				}else{
					v[i]-=len;
					ans[i].PB('U');
				}
			}
		}
		len=max(len/2,int(1));
	}
	cerr<<d<<endl;
	cerr<<ans<<endl;
	Validate(rawX,rawY,d,ans);
	const int m=d.size();
	print(m);
	REP(i,m)
		print(d[i],i==m-1?1:2);
	REP(i,n)
		cout<<ans[i]<<endl;
	cerr<<"Score: "<<m<<endl;
}

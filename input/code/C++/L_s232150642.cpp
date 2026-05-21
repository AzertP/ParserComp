#include<iostream>
#include<algorithm>
#include<set>
#include<queue>
using namespace std;
#define rep(i,n) for(int i=0;i<n;i++)
#define rp(i,c) for(int i=0;i<(c).size();i++)
#define fr(i,c) for(__typeof((c).begin()) i=(c).begin();i!=(c).begin();i++)
#define mp make_pair
#define pb push_back

typedef vector<int> vi;
typedef pair<int,int> pi;

const int inf=1<<28;
const int dy[]={-1,-1,-1,0,1,1,1,0},dx[]={-1,0,1,1,1,0,-1,-1};

int n;

int buftoI(char buf[6][6])
{
	int y,x;
	rep(i,n)rep(j,n)if(buf[i][j]=='@')y=i,x=j;
	
	int ret=(y<<3)+x;
	rep(i,n)rep(j,n)ret*=2,ret+=(buf[i][j]=='#');
	return ret;
}
pi ItoBuf(int s,char buf[6][6])
{
	for(int i=n-1;i>=0;i--)for(int j=n-1;j>=0;j--)
	{
		int t=s&1;
		buf[i][j]=t?'#':'.';
		s/=2;
	}
	int x=s&7,y=s>>3;
	buf[y][x]='@';
	return mp(y,x);
}
void moveTime(char buf[6][6])
{
	char tmp[6][6];
	
	rep(i,n)rep(j,n)
	{
		int cnt=0;
		rep(d,8)
		{
			int ny=i+dy[d],nx=j+dx[d];
			if(ny<0||nx<0||ny>=n||nx>=n)continue;
			if(buf[ny][nx]!='.')cnt++;
		}
		if(buf[i][j]=='#')
		{
			if(cnt==2||cnt==3)tmp[i][j]='#';
			else tmp[i][j]='.';
		}
		else if(buf[i][j]=='.')
		{
			if(cnt==3)tmp[i][j]='#';
			else tmp[i][j]='.';
		}
		else tmp[i][j]='@';
	}
	
	rep(i,n)rep(j,n)buf[i][j]=tmp[i][j];
}

int main()
{

	while(cin>>n,n)
	{
		char buf[6][6];
		rep(i,n)cin>>buf[i];
		//BFS
		int ans=-1,cur=buftoI(buf);
		queue<pi> Q;
		Q.push(mp(0,cur));
		set<int> V;
		V.insert(cur);
		while(!Q.empty())
		{
			int step=Q.front().first;
			cur=Q.front().second;
			Q.pop();
			
			if((cur&((1<<n*n)-1))==0)
			{
				ans=step;break;
			}
			
			pi p=ItoBuf(cur,buf);
			int y=p.first,x=p.second;
			
			rep(d,8)
			{
				int ny=y+dy[d],nx=x+dx[d];
				if(ny<0||nx<0||ny>=n||nx>=n)continue;
				if(buf[ny][nx]!='.')continue;
				
				char nxtbuf[6][6];
				rep(i,n)rep(j,n)nxtbuf[i][j]=buf[i][j];		
				nxtbuf[ny][nx]='@',nxtbuf[y][x]='.';
				moveTime(nxtbuf);
				int nxt=buftoI(nxtbuf);
				
				if(V.count(nxt))continue;
				V.insert(nxt);
				Q.push(mp(step+1,nxt));
			}
		}
		cout<<ans<<endl;
	}
	return 0;
}
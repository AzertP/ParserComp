#include<stdio.h>
#include<stdlib.h>
#define ll long long
#define rep(i,l,r)for(int i=(l);i<(r);i++)
#define min(p,q)((p)<(q)?(p):(q))
#define INF ((1LL<<62)-(1LL<<31))

int n,m;
char s[110][110];
int d4[5]={0,1,0,-1};

//辺の情報を個別に持つタイプ
typedef struct edge{ll s,g,c;}E;
typedef struct graph{
	int vcnt,ecnt;
	E  e[200010];//適宜変える(ecnt)
	int inv[200010];//逆辺のindex(ecnt)
	int id[10010];//適宜変える(vcnt)
}G;
G g;

int esort(const void*a,const void*b){
	E*p=(E*)a,*q=(E*)b;
	if((*p).s<(*q).s)return -1;
	if((*p).s>(*q).s)return  1;
	if((*p).g<(*q).g)return -1;
	return 1;
}
void makeinvedge();
void readgraph(){
	//x+yが偶数から奇数へ辺をはる
	int ecnt=0;
	rep(x,0,n)rep(y,0,m){
		if(s[x][y]=='#')continue;
		if((x+y)%2==0){
			rep(k,0,4){
				int xx=x+d4[k];
				int yy=y+d4[k+1];
				if(0<=xx&&xx<n&&0<=yy&&yy<m){
					g.e[ecnt].s= x*m+ y;
					g.e[ecnt].g=xx*m+yy;
					g.e[ecnt].c=1;
					ecnt++;
				}
			}
			g.e[ecnt].s=n*m;
			g.e[ecnt].g=x*m+y;
			g.e[ecnt].c=1;
			ecnt++;
		}else{
			g.e[ecnt].s=x*m+y;
			g.e[ecnt].g=n*m+1;
			g.e[ecnt].c=1;
			ecnt++;
		}
	}
	
	g.vcnt=n*m+2;
	g.ecnt=ecnt;
	qsort(g.e,g.ecnt,sizeof(E),esort);
	makeinvedge();

	int p=0;
	rep(i,0,g.vcnt){
		while(p<g.ecnt&&g.e[p].s<i)p++;
		g.id[i]=p;
	}
	g.id[g.vcnt]=g.ecnt;//番兵
}


//*
void makeinvedge(){
	//逆辺とidx
	int added=0;
	rep(i,0,g.ecnt){
		int l=0,r=g.ecnt;
		while(r-l>1){
			int m=(l+r)/2;
			if(g.e[m].s<g.e[i].g||(g.e[m].s==g.e[i].g&&g.e[m].g<=g.e[i].s))l=m;
			else r=m;
		}
		if(g.e[l].s!=g.e[i].g||g.e[l].g!=g.e[i].s){
			g.e[g.ecnt+added].s=g.e[i].g;
			g.e[g.ecnt+added].g=g.e[i].s;
			g.e[g.ecnt+added].c=0;
			added++;
		}
	}
	g.ecnt+=added;
	qsort(g.e,g.ecnt,sizeof(E),esort);

	int p=0;
	rep(i,0,g.vcnt){
		while(p<g.ecnt&&g.e[p].s<i)p++;
		g.id[i]=p;
	}
	g.id[g.vcnt]=g.ecnt;//番兵

	rep(i,0,g.ecnt){
		int l=0,r=g.ecnt;
		while(r-l>1){
			int m=(l+r)/2;
			if(g.e[m].s<g.e[i].g||(g.e[m].s==g.e[i].g&&g.e[m].g<=g.e[i].s))l=m;
			else r=m;
		}
		g.inv[i]=l;
	}
}
//*/


//dinic O(VVE)
//ソースs,シンクtを引いてsからtへの最大流を返す
//*
int dist[10010];//ソースからの距離
int checked[10010];//dfsの行き止まりフラグ
void dinicbfs(int s){
	rep(i,0,g.vcnt)dist[i]=-1;
	dist[s]=0;
	//まだ流せる辺だけを使ってbfs
	int que[10010],qcnt=0;
	que[qcnt++]=s;
	rep(q,0,qcnt){
		int v=que[q];
		rep(i,g.id[v],g.id[v+1])if(g.e[i].c>0&&dist[g.e[i].g]==-1){
			dist[g.e[i].g]=dist[v]+1;
			que[qcnt++]=g.e[i].g;
		}
	}
}
ll dinicdfs(int a,int t,ll m){
	//aはm受け取ってる(⇔aから最大m流せる)
	//いくら流せるかを返す
	if(a==t)return m;
	if(checked[a])return 0;
	ll ans=0;
	checked[a]=1;
	rep(i,g.id[a],g.id[a+1])if(g.e[i].c>0&&dist[g.e[i].g]>dist[a]){
		int b=g.e[i].g;
		ll addedflow=dinicdfs(b,t,min(m,g.e[i].c));
		if(addedflow){
			g.e[i].c-=addedflow;
			g.e[g.inv[i]].c+=addedflow;
			checked[a]=0;
			ans+=addedflow;
			m-=addedflow;
			if(m<=0)break;
		}
	}
	return ans;
}
//ソース,シンク
ll dinic(int s,int t){
	ll ans=0;
	int flag=1;
	while(flag){
		flag=0;//更新フラグ
		dinicbfs(s);
		while(!checked[s]){
			ll addedflow=dinicdfs(s,t,INF);
			ans+=addedflow;
			if(addedflow)flag=1;
		}
		rep(i,0,g.vcnt)checked[i]=0;
	}
	return ans;
}
//*/

int main(){
	scanf("%d%d",&n,&m);
	rep(i,0,n)scanf("%s",s[i]);
	readgraph();
	//rep(i,0,g.ecnt)printf("%d %d %d\n",g.e[i].s,g.e[i].g,g.e[i].c);
	printf("%lld\n",dinic(n*m,n*m+1));
	rep(i,0,g.ecnt)if(g.e[i].c==0){
		int sx=g.e[i].s/m;
		int sy=g.e[i].s%m;
		int gx=g.e[i].g/m;
		int gy=g.e[i].g%m;
		if(sx==n||gx==n)continue;
		if((sx+sy)%2!=0)continue;
		
		//気合で出力文字を場合分け
		if(sx==gx){
			if(sy<gy){
				s[sx][sy]='>';
				s[gx][gy]='<';
			}else{
				s[sx][sy]='<';
				s[gx][gy]='>';
			}
		}else{
			if(sx<gx){
				s[sx][sy]='v';
				s[gx][gy]='^';
			}else{
				s[sx][sy]='^';
				s[gx][gy]='v';
			}
		}
	}
	rep(i,0,n)puts(s[i]);
}

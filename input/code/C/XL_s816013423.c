#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<stdbool.h>
#include<assert.h>
typedef long long ll;
typedef long double ld;
#define rep(i,l,r)for(ll i=(l);i<(r);i++)
#define repp(i,l,r,k)for(ll i=(l);i<(r);i+=(k))
#define rrep(i,l,r)for(ll i=(l);i>=(r);i--)
#define INF (1LL<<60)
#define MOD1 1000000007
#define MOD2 998244353
#define MAX_N (1 << 17)
#define YES printf("Yes\n")
#define NO printf("No\n")
#define PN printf("\n")
#define charsize 100005 //10^5+5

void swap(ll *a, ll *b){ll c;c=*b;*b=*a;*a= c;}
// void fill(ll n, ll a[n], ll value){ rep(i,0,n) a[i] = value; }
void cin(ll *n){ scanf("%lld",&(*n)); }
void cin2(ll *a, ll *b){ scanf("%lld%lld",&(*a),&(*b)); }
void cin3(ll *a, ll *b, ll *c){ scanf("%lld%lld%lld",&(*a),&(*b),&(*c)); }
ll max2(ll a,ll b){return a>=b?a:b;}
ll min2(ll a,ll b){return a>=b?b:a;}
ll min3(ll a, ll b, ll c){return (a<=b && a<=c) ? a : b<=c ? b : c;}
ll max3(ll a, ll b, ll c){return (a>=b && a>=c) ? a : b>=c ? b : c;}
ll minn(ll n, ll a[n]){ll b=INF;rep(i,0,n) b=min2(b,a[i]);return b;}
ll maxn(ll n, ll a[n]){ll b=-INF;rep(i,0,n) b=max2(b,a[i]);return b;}
ll ABS(ll a){return a>=0?a:(-a);}
ll POW(ll a, ll b){ll c=1;rep(i,0,b) c*=a;return c;}
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll lcm(ll a,ll b){return a/gcd(a,b)*b;}
ll mod_MOD1(ll n){n+= n<0?((-n)/MOD1+1)*MOD1:0; return n%=MOD1;}
ll mod_p(ll n ,ll p){n+= n<0?((-n)/p+1)*p:0; return n%=p;}
ll change_into_num(char s[] , ll len, ll p){ return !p ? 0 : POW(10,p-1)*(s[len-p]-'0') + change_into_num(s,len,p-1); }

int upll(const void*a, const void*b){return*(ll*)a<*(ll*)b?-1:*(ll*)a>*(ll*)b?1:0;}
int downll(const void*a, const void*b){return*(ll*)a<*(ll*)b?1:*(ll*)a>*(ll*)b?-1:0;}
void sortup(ll*a,int n){qsort(a,n,sizeof(ll),upll);}
void sortdown(ll*a,int n){qsort(a,n,sizeof(ll),downll);}

typedef struct{ ll a , b;}fr;
int cmp1( const void *p, const void *q ) { return ((fr*)p) ->a - ((fr*)q)->a;}
int cmp2( const void *p, const void *q ) { return ((fr*)q) ->a - ((fr*)p)->a;}
void strsortup(fr*a,int n){qsort(a,n,sizeof(fr),cmp1);}
void strsortdown(fr*a,int n){qsort(a,n,sizeof(fr),cmp2);}

/*------------------------------Dijkstra start-------------------------------------------------------------*/
//http://www.deqnotes.net/acmicpc/dijkstra/
int edge_cnt[114123];
ll* graph[114123];
ll ans[114123];
bool is_conf[114123];
// freeeeeeeeeeeeeeeeeeeeeeeeeeee!!!!!!!!!!!!

void pushGraph(ll v1, ll v2, ll weight , ll is_dir , ll snuke){
    graph[v1] = realloc( graph[v1] , sizeof(ll)*(edge_cnt[v1]*3+3) );    
    graph[v1][edge_cnt[v1]*3] = v2;
    graph[v1][edge_cnt[v1]*3+1] = weight;
    graph[v1][edge_cnt[v1]*3+2] = snuke;
    // ednum[v1] = realloc( ednum[v1] , sizeof(ll)*(edge_cnt[v1]+1) );
    // ednum[v1][edge_cnt[v1]] = num;
    edge_cnt[v1]++;

    if(!is_dir){
        graph[v2] = realloc( graph[v2] , sizeof(ll)*(edge_cnt[v2]*3+3) );
        graph[v2][edge_cnt[v2]*3] = v1;
        graph[v2][edge_cnt[v2]*3+1] = weight;
        graph[v2][edge_cnt[v2]*3+2] = snuke;
        // ednum[v2] = realloc( ednum[v2] , sizeof(ll)*(edge_cnt[v2]+1) );
        // ednum[v2][edge_cnt[v2]] = num;
        edge_cnt[v2]++;
    }
}
/*-----------priorityqueue(strurct)---------------*/
// kind = max?1:0
ll HeapValue(ll a){return a?-INF:INF;}
ll Heapcmp(ll a, ll b, ll kind){return kind ? (a<b?1:0) : (a>b?1:0);}
typedef struct{ int from; int to; ll cost; }graph_heap;
graph_heap heap[2*MAX_N];
void str_swap(graph_heap *a,graph_heap *b){ graph_heap c;c=*b;*b=*a;*a=c; }

void graph_pushHeap(ll val, ll len , ll kind){
    ll i=len; heap[i].cost=val;
    while(i!=0){
        if( Heapcmp( heap[i].cost, heap[(i-1)/2].cost , ABS(kind-1) )  ) str_swap(&heap[i],&heap[(i-1)/2]);
        i--;i/=2;
    }
}
void graph_popHeap(ll len, ll kind){
    ll i=0; heap[0].cost=HeapValue(kind);
    str_swap(&heap[0],&heap[len-1]);
    while(len>i*2+1){
        if(i*2+2<len && Heapcmp(heap[i].cost,heap[i*2+2].cost,kind) && Heapcmp(heap[i*2+1].cost,heap[i*2+2].cost,kind) ){
            str_swap(&heap[i],&heap[i*2+2]);
            i*=2;i+=2;
        }else{
            if( Heapcmp(heap[i].cost,heap[i*2+1].cost,kind) ){
                str_swap(&heap[i],&heap[i*2+1]);
                i*=2;i++;
            }else break;
        }
    }
}
/*---------------------------------------------*/

void dijkstra(ll start, ll snuke){
    rep(i,0,114123){
        ans[i] = INF;
        is_conf[i] = false;
    }
    ans[start] = 0;
    is_conf[start]  = true;
    int size=0;
    rep(i,0,edge_cnt[start]){
        if( snuke != graph[start][i*3+2] ) continue;
        heap[size].from = start;
        heap[size].to = graph[start][i*3];
        ans[ heap[size].to ] = graph[start][i*3+1];
        graph_pushHeap( graph[start][i*3+1] , size , 0 );
        size++;        
    }
    // PN;
    // rep(i,0,4){
    //     rep(j,0,edge_cnt[i]){
    //         printf("%lld %lld %lld %lld\n",i,graph[i][j*3],graph[i][j*3+1],graph[i][j*3+2] );
    //     }
    // }
    // PN;
    // rep(i,0,4) printf("%lld ",ans[i]);
    // PN;
    // ll piv;
    while(size){
        // printf("size:%d\n",size);
        if( is_conf[ heap[0].to ] ){
            graph_popHeap(size,0);
            size--;
            continue;
        }
        // printf("AA: %d %d\n",heap[0].from, heap[0].to );
        is_conf[ heap[0].to ] = true;
        ans[ heap[0].to ] = heap[0].cost;
        // piv = heap[0].to;
        // rep(i,0,4) printf("%lld ",ans[i]);
        // PN;
        rep(i,0,edge_cnt[ heap[0].to ]){
            // printf("%lld\n",graph[ heap[0].to ][i*3+2] );
            if( snuke != graph[ heap[0].to ][i*3+2] ) continue;
            if( ans[ graph[ heap[0].to ][i*3] ] > ans[ heap[0].to ] + graph[ heap[0].to ][i*3+1] ){
                heap[size].from = heap[0].to;
                heap[size].to = graph[ heap[0].to ][i*3];
                // printf("%d %d %lld\n",heap[size].from,heap[size].to, ans[ heap[0].to ] + graph[ heap[0].to ][i*3+1] );
                graph_pushHeap( ans[ heap[0].to ] + graph[ heap[0].to ][i*3+1] , size , 0 );
                size++;
            }
        }
        graph_popHeap(size,0);
        size--;
    }
}
/*---------------------------Dijkstra end--------------------------------------------------------------------*/


/*  ll t=1;
    while(t<n) t*=2;
木の最底辺の大きさを決める

    seg木のsize:2*t-1
*/ 
ll seg[2*MAX_N-1];
void segmake(ll n, ll a[n] ,ll t){
    rep(i,0,t) seg[t-1+i]=i<n?a[i]:INF;
    while(t/=2) rep(i,0,t) seg[t+i-1]=min2(seg[2*(t+i)],seg[2*(t+i)-1]); 
}
// 初期状態 segmin( x[i] , y[i] , 0 , 0 , t-1 )
ll segmin(ll from, ll to, ll a, ll l, ll r){
    return (l>=from && r<=to) ? seg[a] : (l>to || r<from) ? INF : min2(segmin(from,to,2*a+1,l,(l+r)/2),segmin(from,to,2*a+2,(l+r)/2+1,r));
}
ll seggcd(ll from, ll to, ll a, ll l, ll r){
    return (l>=from && r<=to) ? seg[a] : (to<(l+r)/2+1) ? seggcd(from,to,2*a+1,l,(l+r)/2) : (from>(l+r)/2) ? seggcd(from,to,2*a+2,(l+r)/2+1,r) : gcd( seggcd(from,to,2*a+2,(l+r)/2+1,r) , seggcd(from,to,2*a+1,l,(l+r)/2) );
}

void segupdate(ll a, ll b, ll t){
    ll c=t-1+a;
    seg[c]=b;
    while(c>0){
        c--; c/=2;
        seg[c]=min2(seg[2*(c+1)],seg[2*(c+1)-1]);
    }
}

int main(void){
    // char ;
    ll n,m,s,t;
    // ll ans=0;
    cin(&n);
    cin(&m);
    cin2(&s,&t);
    // cin3(&,&,&);
    ll u,v,a,b;
    rep(i,0,m){
        cin2(&u,&v);
        cin2(&a,&b);
        pushGraph(u-1,v-1,a,0,0);
        pushGraph(u-1,v-1,b,0,1);
    }
    // PN;
    // rep(i,0,n){
    //     rep(j,0,edge_cnt[i]){
    //         printf("%lld %lld %lld %lld\n",i,graph[i][j*3],graph[i][j*3+1],graph[i][j*3+2] );
    //     }
    // }
    // PN;
    // PN;
    ll total_cost[n];
    rep(i,0,n) total_cost[i]=0;
    dijkstra(s-1,0);
    // rep(i,0,n) printf("%lld ",ans[i] );
    // PN;
    rep(i,0,n) total_cost[i]+=ans[i];
    dijkstra(t-1,1);
    // rep(i,0,n) printf("%lld ",ans[i] );
    // PN;
    rep(i,0,n) total_cost[i]+=ans[i];
    // rep(i,0,n) printf("%lld ",total_cost[i] );
    // PN;

    ll treesize=1;
    while(treesize<n) treesize*=2;
    segmake(n,total_cost,treesize);

    ll kenko = POW(10,15);
    rep(i,0,n){
        printf("%lld\n", kenko- segmin(i,n-1,0,0,treesize-1) );
    }

    
    // printf("%lld\n");
    return 0;
}

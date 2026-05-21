#include <stdio.h>
#define INF 100000000

int n,G[103][103];

int prim(void);

int main()
{
  int i,j;

  scanf("%d",&n);

  /* 重みつきグラフが与えられる。辺がないときは-1 */
  for(i = 1;i <= n;i++){
    for(j = 1;j <= n;j++){
      scanf("%d",&G[i][j]);
    }
  }

  /* 辺がないときをInfにする */
   for(i = 1;i <= n;i++){
    for(j = 1;j <= n;j++){
      if(G[i][j] == -1) G[i][j] = INF;
    }
   }

   printf("%d\n",prim());
   
   return 0;

}

int prim(void)
{
  int u,v,i,mincost;
  int d[103],pi[103];
  char color[103];
  int total = 0;

  /* 初期化 */
  for(u = 1;u <= n;u++){
    d[u] = INF;
    pi[u] = -2;
    color[u] = 'W';
  }

  d[1] = 0;
    
  while(1){
    mincost = INF;
    for(i = 1;i <= n;i++){
      if(color[i] != 'B' && d[i] < mincost){
	mincost = d[i];
	u = i;
      }
    }
    
    if(mincost == INF) break;

    color[u] = 'B';
    
    for(v = 1;v <= n;v++){
      if(color[v] != 'B' && G[u][v] < d[v]){
	pi[v] = u;
	d[v] = G[u][v];
      }
    }
    color[v] = 'B';
  }

  for(i = 1;i <= n; i++){
    if(pi[i] != -2) total += G[i][pi[i]];
  }

  return total;
}
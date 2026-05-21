#include <stdio.h>

#define WHITE 1
#define GRAY 0
#define BLACK -1
#define INF 100000000

void bfs(int);
void enqueue(int);
int dequeue(void);

int Q[102],d[102],color[102],time,n,vv[102][102],k[102];
int head, tail;

int main()
{
  int i,j,u,min;

  scanf("%d",&n);

  head = 0;
  tail = 1;

  for(i=0;i<n;i++){
    scanf("%d",&u);
    scanf("%d",&k[u]);

    for(j=0;j<k[u];j++){
      scanf("%d",&vv[u][j]);
    }
  }

  bfs(1);
  for(i=1;i<=n;i++)  {
    if(d[i]==INF) d[i]=-1;
    printf("%d %d\n",i,d[i]);
  }

  return 0;
}

void bfs(int s){
  int i,u,v;
  for (u=1;u<=n;u++){
    color[u] = WHITE;
    d[u] = INF;
  }
  color[s] = GRAY;
  d[s] = 0;
  // Q = empty;
  enqueue(s);

  while(head!=tail){
    u = dequeue();
    for(i=0;i<k[u];i++){
      v=vv[u][i];
      if (color[v] == WHITE){
	color[v] = GRAY;
	d[v] = d[u] + 1;
	enqueue(v);
      }
      color[u] = BLACK;
    }
  }
}

void enqueue(int x){ /* キューの末尾に要素 x を追加 */
  Q[tail] = x;
  if (tail + 1 == n) tail = 0;
  else tail++;
}

int dequeue(void){ /* キューの先頭から要素を取り出す */
  int x;
  x = Q[head];
  if (head + 1 == n) head = 0;
  else head++;
  return x;
}
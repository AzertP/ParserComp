#include<stdio.h>
#include<stdlib.h>
struct node{
  int key;
  struct node *next;
};
typedef struct node *GG;

GG G[100000];
int color[100000];

GG MNode(int,GG);
void Insert(int,int);
void Dye(int,int);
int main()
{
  int p,i,s,t,id=1,q,n,m;
  scanf("%d%d",&n,&m);
  for(i=0;i<n;i++)G[i]=MNode(i,NULL);
  for(i=0;i<m;i++){
    scanf("%d%d",&s,&t);
    Insert(s,t);
    Insert(t,s);
  }
  for(i=0;i<n;i++)color[i]=0;
  for(i=0;i<n;i++){
    if(color[i]==0){
      Dye(i,id);
      id++;
    }
  }
  
  scanf("%d",&q);
  for(i=0;i<q;i++){
    scanf("%d%d",&s,&t);
    if(color[s]==color[t])printf("yes\n");
    else printf("no\n");
  }
  // for(i=0;i<n;i++)printf("%d ",color[i]);
  return 0;
}

GG MNode(int key,GG Node)
{
  GG n;
  n=(GG)malloc(sizeof(struct node));
  n->key=key;
  n->next=Node;
  return n;
}
void Insert(int s,int t)
{
  GG GGG;
  GGG=MNode(t,G[s]->next);
  G[s]->next=GGG;
}

void Dye(int c,int id)
{
  int S[100000],head=0,u,v;
  GG GGG;
  S[head]=c;
  head++;
  color[c]=id;
  while(head!=0){
    u=S[head-1];
    head--;
    for(GGG=G[u]->next;GGG!=NULL;GGG=GGG->next){
      v=GGG->key;
      if(color[v]==0){
        color[v]=id;
	S[head]=v;
	head++;
      }
    }
  }
}
    
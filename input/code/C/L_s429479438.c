#include<stdio.h>
#include<stdlib.h>
struct node{
  int key;
  struct node *r,*l,*p;
};
struct node *rt;
void insert(int);
void in(struct node *);
void pre(struct node *);
struct node *find(struct node *,int);
int main(){
  int n,i,x;
  struct node *y;
  char c[7];
  scanf("%d",&n);
  i=0;
  while(i<n){
    scanf("%s",c);
    if(c[0]=='i'){
       scanf("%d",&x);
       insert(x);
    }
    else if(c[0]=='p'){
      in(rt);
      printf("\n");
      pre(rt);
      printf("\n");
    }
    else if(c[0]=='f'){
       scanf("%d",&x);
       y=find(rt,x);
       if(y!=NULL)printf("yes\n");
       else printf("no\n");
    }
    i++;
  }
  return 0;
}

void insert(int k){
  
  struct node *x=rt,*y=NULL,*z;

  z=(struct node *)malloc(sizeof(struct node));
  z->key=k; z->l=NULL; z->r=NULL;
  while(x!=NULL){
    y=x;
    if(z->key < x->key)x=x->l;
    else x=x->r;
  }
    z->p=y;
    
    if(y==NULL)rt=z;
    else {
      if(z->key < y->key) y->l=z;
      else y->r=z;
    
  }
}


void pre(struct node *p){

  if(p==NULL) return 0;
  printf(" %d",p->key);
  pre(p->l);
  pre(p->r);
}

void in(struct node *p){
  
  if(p==NULL) return 0;
   in(p->l);
   printf(" %d",p->key);
   in(p->r);
}

struct node *find(struct node *p,int k){
 
  while(p!=NULL&&k!=p->key){
    if(k < p->key)p=p->l;
    else p=p->r;
  }
  return p;
  
}


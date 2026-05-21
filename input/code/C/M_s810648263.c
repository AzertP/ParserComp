#include<stdio.h>
#include<string.h>
#define LEN 100005
  
typedef struct kouzou{
  char name[100];
  int t;
}K;
  
K Q[LEN];
int head,tail,n;
  
void enqueue(K x){
  Q[tail] = x;
  tail += 1;
  tail = tail % LEN;
}
  
K dequeue(){
  K x = Q[head];
  head += 1;
  head = head % LEN;
  return x;
}
  
int main(void){
  int min(int,int);
  int m = 0,a,i,q;
  K u;
  scanf("%d%d",&n,&q);
  for ( i = 1 ; i <= n ; i++ ){
    scanf("%s",Q[i].name);
    scanf("%d",&Q[i].t);
  }
  head = 1;
  tail = n + 1;
  while ( head != tail ){
    u = dequeue();
    a = min(q,u.t);
    u.t -= a;
    m += a;
    if ( u.t > 0 ){
      enqueue(u);
    }
    else{
      printf("%s %d\n",u.name,m);
    }
  }
  return 0;
}
  
int min(int b, int c ){
  return b < c ? b:c;
}
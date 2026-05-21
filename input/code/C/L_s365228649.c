#include<stdio.h>
#include<string.h>

#define LEN 100005

typedef struct pp{
  char name[100];
  int t;
}P;

void initialize(void);
int isEmpty(void);
int isFull(void);
void enqueue(P);
P dequeue(void);

P Q[LEN+1];
int head, tail, n;

void initialize(){
  head = 0;
  tail = 0;
}
int isEmpty(){
  if(head == tail)
    return 1;
  else return 0;
}

int isFull(){
  if(head == (tail + 1) % LEN)
    return 1;
  else return 0;
}

void enqueue(P x){
  if(isFull()==1){
     fprintf(stderr,"Over\n");
  }
  Q[tail]=x;
  if((tail+1)==LEN){
    tail=0;
  }
  else tail++;
}

P dequeue(){
  P x;
  if(isEmpty()==1){
    fprintf(stderr,"Under\n");
  }
  x = Q[head];
  if((head+1)== LEN){
    head=0;
  }
  else head++;
  return x;
}


int main(){
  int elaps = 0,c=0;
  int i, q;
  P u;
  scanf("%d %d", &n, &q);
  head = n;
  tail = n+1;
  for ( i = 1; i <= n; i++){
    scanf("%s", Q[i].name);
    scanf("%d", &Q[i].t);
  }
  initialize();
  for(i=1;i<=n;i++){
    enqueue(Q[i]);
  }
  while(c!=n){
    u=Q[head];
    u.t=Q[head].t-q;
    if(u.t<=0){
      elaps+=u.t+q;
      dequeue();
      printf("%s %d\n",u.name,elaps);
      c++;
    }
    else{
      elaps+=q;
      dequeue();
      enqueue(u);
    }
  }
  
  
  return 0;
}


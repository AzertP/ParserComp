#include <stdio.h>
#include <stdlib.h>
#define MAX 200

typedef struct{
  int top;
  int number[MAX];
} STACK;

void initialize(STACK *);
void push(STACK *, int);
int pop(STACK *);
int isFull(STACK *);
int isEmpty(STACK *);

int main(){
  STACK stack, *s;
  char str[200];
  int x, y;
  s = &stack;
  initialize(s);

  while(scanf("%s",str) != EOF){
    if(str[0] == '+'){
      x = pop(s) + pop(s);
      push(s,x);
    }
    else if(str[0] == '-'){
      x = pop(s);
      y = pop(s);
      x = y - x;
      push(s,x);
    }
    else if(str[0] == '*'){
      x = pop(s) * pop(s);
      push(s,x);
    }
    else if(str[0] <= '9' && str[0] >= '0'){
      x = atoi(str);
      push(s,x);
    }
    else break;
  }
  x = pop(s);
  printf("%d\n",x);
  return 0;
}

void initialize(STACK *s){
  s->top = 0;
}
void push(STACK *s, int x){
  if(isFull(s) == 1) exit(1);
  else{
    s->top += 1;
    s->number[s->top] = x;
  }
}

int pop(STACK *s){
  if(isEmpty(s) == 1) exit(2);
  else{
    s->top -= 1;
    return s->number[s->top+1];
  }
}

int isEmpty(STACK *s){
  if(s->top == 0) return 1;
  else return 0;
}

int isFull(STACK *s){
  if(s->top >= MAX-1) return 1;
  else return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
struct node{
  int key;
  struct node *next, *prev;
};
 
typedef struct node *NodePointer;
 
void insert(int);
void deleteNode(NodePointer);
void delete(int);
void init(void);
void deletefirst(void);
void deletelast(void);
NodePointer finditem(int);
void listprint(void);
 
NodePointer head;
 
int main(){
  int n, i, b;
  char c[15];
 
  scanf("%d", &n);
  init();
  for(i = 0;i < n; i++){
    scanf("%s%d", c, &b);
    if(strcmp(c, "insert") == 0){
      insert(b);
    }
    else if(strcmp(c, "delete") == 0){
      delete(b);
    }
    else if(strcmp(c, "deleteFirst") == 0){
      deletefirst();
    }
    else if(strcmp(c, "deleteLast") == 0){
      deletelast();
    }
  }
  listprint();
  return 0;
}
 
void insert(int data){
  NodePointer x = (NodePointer)malloc(sizeof(NodePointer));
 
  x->key = data;
  x->next = head->next;
  head->next->prev = x;
  head->next = x;
  x->prev = head;
}
 
void listprint(){
  NodePointer n = head->next;
  int flg = 0;
 
  while(1){
    if(n == head) break;
    if(flg++ > 0) printf(" ");
    printf("%d", n->key);
    n = n->next;
  }
  printf("\n");
}
 
NodePointer finditem(int data){
  NodePointer n = head->next;
 
  while(n != head && n->key != data){
    n = n->next;
  }
  return n;
}
 
void init(){
  head = (NodePointer)malloc(sizeof(NodePointer));
  head->next = head;
  head->prev = head;
}
 
void deleteNode(NodePointer a){
  if(a == head) return;
  a->prev->next = a->next;
  a->next->prev = a->prev;
  free(a);
}
 
void delete(int keydata){
  deleteNode(finditem(keydata));
}
 
void deletefirst(){ 
  deleteNode(head->next);
}
 
void deletelast(){
  deleteNode(head->prev);
}
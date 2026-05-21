#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node{
  int key;
  struct node *prev;
  struct node *next;
};

typedef struct node node;

void insertX(int);
int deleteX(int);
void deleteFirst(void);
void deleteLast(void);

node *sentinel; /* 番兵 */


int main() {

  int n;
  int inputKey;
  char inputOrder[12];

  node *printNode;

  int i;

  sentinel = (node *)malloc( sizeof( node ) );
  sentinel->next = sentinel;
  sentinel->prev = sentinel;


  scanf("%d", &n);

  for( i=0; i<n; i++) {
    scanf("%s", inputOrder);
    if( strcmp( inputOrder, "deleteFirst") == 0 ) {
      deleteFirst();
    }
    else if( strcmp( inputOrder, "deleteLast") == 0 ) {
      deleteLast();
    }
    else {
      scanf("%d", &inputKey);
      if( strcmp( inputOrder, "delete") == 0 ) {
	deleteX(inputKey);
      }
      else {
	insertX(inputKey);
      }
    }
  }

  for( printNode = sentinel->next; printNode->next != sentinel; printNode = printNode->next) {
    printf("%d ", printNode->key);
  }
  printf("%d\n", printNode->key);

  return 0;
}


void insertX(int x) {
  
  node *insNode;

  insNode = (node *)malloc( sizeof(node) );

  insNode->next = sentinel->next;
  sentinel->next->prev = insNode;
  sentinel->next = insNode;
  insNode->prev = sentinel;

  insNode->key = x;

}


int deleteX(int x) {
  
  node *delNode;

  /* deleteKeyと一致するノードをさがす */
  for( delNode=sentinel->next; delNode!=sentinel; delNode=delNode->next) {
    if(delNode->key == x) {
      break;
    }
  }
  if(delNode==sentinel){ /*該当するノードは存在しない */
    return 0;
  }

  /* ノードを削除する */
  delNode->prev->next = delNode->next;
  delNode->next->prev = delNode->prev;

  return 1;

}
  
void deleteFirst() {
  
  sentinel->next->next->prev = sentinel;
  sentinel->next = sentinel->next->next;

}


void deleteLast() {
  
  sentinel->prev->prev->next = sentinel;
  sentinel->prev = sentinel->prev->prev;

}
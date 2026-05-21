#include <stdio.h>
#include <stdlib.h>

typedef struct node{
  struct node *left;
  struct node *right;
  struct node *parent;
  int key;
}Node;
typedef Node *NodePointer;

NodePointer createNode(void);
void inorder(NodePointer);
void preorder(NodePointer);
void insert(int);
NodePointer find(int, NodePointer);
NodePointer root = NULL;

int main(){

  int i, n, x;
  char str[10];

  scanf("%d", &n);
  for(i = 0; i < n; i++){
    scanf("%s", str);
    if(str[0] == 'i'){
      scanf("%d", &x);
      insert(x);
    }else if(str[0] == 'p'){
      inorder(root);
      printf("\n");
      preorder(root);
      printf("\n");
    }else if(str[0] == 'f'){
      scanf("%d", &x);
      if(find( x, root) == NULL){
	printf("no\n");
      }else{
	printf("yes\n");
      }
    }else{
      continue;
    }
  }

  return 0;
}

NodePointer createNode(void){

  NodePointer tree = malloc(sizeof(Node));
  tree->left = NULL;
  tree->right = NULL;
  tree->parent = NULL;
  tree->key = 0;

  return tree;
}

void insert(int key){

  NodePointer z = createNode();
  NodePointer x = root;
  NodePointer y = NULL;
  z->key = key;
  while(x != NULL){
    y = x;

    if(z->key < x->key){
      x = x->left;
    }else{
      x = x->right;
    }
  }

  z->parent = y;

  if(y == NULL){
    root = z;
  }else if(z->key < y->key){
    y->left = z;
  }else{
    y->right = z;
  }
}

void inorder(NodePointer T){

  if(T != NULL){
    inorder(T->left);
    printf(" %d",T->key);
    inorder(T->right);
  }
}

void preorder(NodePointer T){

  if(T != NULL){
  printf(" %d", T->key);
    preorder(T->left);
    preorder(T->right);
  }
}

NodePointer find(int k, NodePointer T){

  while(T != NULL && T->key != k){
    if(T->key > k){
      T = T->left;
    }else{
      T = T->right;
    }
  }
  return T;
}
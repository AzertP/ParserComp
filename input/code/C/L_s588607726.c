#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node{
  struct node *right;
  struct node *left;
  struct node *parent;
  int key;
};

typedef struct node *Node;

void insert(int);
 int find(int);
void delete(int);
void inParse(Node);
void preParse(Node);
Node Minimum(Node);
Node Successor(Node);
Node findNode(int);

Node root;

int main(){

   int i;
   int n;
   int key;
   int judge;
  char str[7];

  scanf("%d",&n);

  for(i=0;i<n;i++){
  scanf("%s",str);
  if(strcmp(str,"insert")==0){
    scanf("%d",&key);
    insert(key);
  }
  
  else if(strcmp(str,"find")==0){
    scanf("%d",&key);
    judge = find(key);
    if(judge==1) printf("yes\n");
    else printf("no\n");
  }

  else if(strcmp(str,"delete")==0){
    scanf("%d",&key);
    delete(key);
  }
  
  else if(strcmp(str,"print")==0){
     inParse(root);
     printf("\n");
     preParse(root);
     printf("\n");
   }
  }

  return 0;
}

void insert(int k){

  Node x = root;
  Node y = NULL;
  Node z;

  z = malloc(sizeof(struct node));

  z->left  = NULL;
  z->right = NULL;
  z->key = k;

  while(x!=NULL){
    y = x;
    if(z->key < x->key) x = x->left;
    else x = x->right;
    z->parent = y;
  }
  if(y==NULL) root = z;
  else if(z->key < y->key) y->left = z;
  else y->right = z;
}

int find(int k){
  
  Node x = root;
  while(x!=NULL){
   if(k == x->key) return 1;
  else{
    if(k < x->key) x = x->left;
    else x = x->right;
   }
  }
  return -1;
}

Node findNode(int k){
  
  Node x = root;
  while(x!=NULL){
   if(k == x->key) return x;
  else{
    if(k < x->key) x = x->left;
    else x = x->right;
   }
  }
  return NULL;
}

void delete(int k){

  Node x = root;
  Node y = NULL;
  Node z;

  z = findNode(k);

  if(z->left ==NULL || z->right == NULL){
    y = z;
  }
  else{
    y = Successor(z);
  }
  if(y->left != NULL) x = y->left;
  else x = y->right;

  if(x != NULL) x->parent = y->parent;

  if(y->parent == NULL){
    root = x;
    }
    else if(y == y->parent->left){
      y->parent->left = x;
    }
    else{
      y->parent->right = x;
    }
    if(y != z){
      z->key = y->key;
    }
}

void inParse(Node n){
  if(n==NULL) return;  
  inParse(n->left);
  printf(" %d",n->key);
  inParse(n->right);
}


void preParse(Node n){
  if(n==NULL) return;
  printf(" %d",n->key);
  preParse(n->left);
  preParse(n->right);
}

Node Minimum(Node n){
  while(n->left!=NULL){
    n = n->left;
  }
  return n;
}

Node Successor(Node n){

  Node p;
  
  if(n->right != NULL)
  return Minimum(n->right);

  p = n->parent;
  p = p->parent;
  return p;
}
  
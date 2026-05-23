
struct node{
  struct node *right;
  struct node *left;
  struct node *parent;
  int key;
};

typedef struct node *Node;

void insert(int);
 int find(int);
void inParse(Node);
void preParse(Node);

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
  
  if(strcmp(str,"find")==0){
    scanf("%d",&key);
    judge = find(key);
    if(judge==1) printf("yes\n");
    else printf("no\n");
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

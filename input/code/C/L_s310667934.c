struct Node{
  int key;
  struct Node *left;
  struct Node *right;  
  struct Node *p;
};

typedef struct Node *node; 

void insert(node);
void inorder(node);
void preorder(node);
node root=NULL,n[500000];
int main()
{
  int i,num,a; 
  char s[10];
  root= (node)malloc(sizeof(struct Node));
  scanf("%d",&num);
  for(i=0;i<num;i++){        
    n[i] = (node)malloc(sizeof(struct Node));

scanf("%s",s);    
if(s[0]=='i'){	

scanf("%d",&n[i]->key);		
 n[i]->right=NULL;
      n[i]->left=NULL;

insert(n[i]);

    }
     
 else {

   inorder(n[0]);
   printf("\n");
   preorder(n[0]);
 printf("\n");
}  

}

  return 0; 
}

void insert(node z){
 
 node x,y;     
  y = NULL; 
  x =root;
  

while (x!=NULL){
    y = x ;
    if (z->key < x->key)
      x = x->left; 
    else 
      x = x->right;
  } 
  z->p = y;
  if (y ==NULL) 
    root = z;
  else if (z->key < y->key)
    y->left = z ;
  else 
    y->right = z ;
}

void inorder(node u){
  if (u == NULL)
    return;
  inorder(u->left);
  printf(" %d",u->key);

inorder(u->right);


}

void preorder(node u){
  if (u == NULL)
    return;
  printf(" %d",u->key);
  preorder(u->left);
  preorder(u->right);

}

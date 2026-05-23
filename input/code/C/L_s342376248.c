
struct node{
  int key;
  struct node *next;
  struct node *prev;
};

typedef struct node * NodePointer;

NodePointer nil;

NodePointer listSearch(int key){
  NodePointer cur = nil->next;
  while( cur != nil && cur->key != key ){
    cur = cur->next;
  }
  return cur;
}

void init(){
  nil = malloc(sizeof(struct node));
  nil->next = nil;
  nil->prev = nil;
}

void printList(){
  NodePointer cur = nil->next;
  int isf = 1;
  while(1){
    if ( cur == nil ) break;
    if ( isf == 0)  printf(" ");
    printf("%d", cur->key);
    cur = cur->next;
    isf = 0;
  }
  printf("\n");
}

void deleteNode(NodePointer t){
  t->prev->next = t->next;
  t->next->prev = t->prev;
  //  free(t);
}

void deleteFirst(){
  NodePointer t = nil->next;
  if ( t == nil ) return;
  deleteNode(t);
}

void deleteLast(){
  NodePointer t = nil->prev;
  if ( t == nil ) return;
  deleteNode(t);
}

void delete(int key){
  NodePointer t = listSearch(key);
  if ( t == nil ) return;
  deleteNode(t);
}


void insert(int key){
  NodePointer x;
  x = malloc(sizeof(struct node));
  x->key = key;
  x->next = nil->next;
  nil->next->prev = x;
  nil->next = x;
  x->prev = nil;
}

int main(){
  int key, n, i;
  int size = 0;
  char com[20];
  int np = 0, nd = 0;
  scanf("%d", &n);
  init();
  for ( i = 0; i < n; i++ ){
    scanf("%s%d", com, &key);
    if(!( 0 <= key && key <= 2000000000)){printf("error") ;}
    if ( com[0] == 'i' ) {insert(key); np++; size++;}
    else if ( com[0] == 'd' ) {
      if (strlen(com) > 6){
	if ( com[6] == 'F' ) deleteFirst();
	else if ( com[6] == 'L' ) deleteLast();
      } else {
	delete(key); nd++; 
      }
      size--;
    }
    if ( size > 1000000 ) { printf("error"); }
  }

  
  if(!( np+nd <= 2000000)){printf("error") ;}
  if(!( nd <= 20)){printf("error") ;}
  printList();


  return 0;
}

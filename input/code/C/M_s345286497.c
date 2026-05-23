
typedef struct{
  int parent;
  int left;
  int right;
  int depth;
}Node;

Node T[MAX];
int c[MAX];

int getDepth(int);
void print(int);

int main(){
  int l, n, m, i, j, k;

  scanf("%d", &n);
  
  for(i = 0; i < n; i++){
    T[i].parent = -1;
    T[i].left = -1;
    T[i].right = -1;
  }
  
  for(i = 0; i < n; i++){
    scanf("%d%d", &l, &m);
    for(j = 0; j < m; j++){
      scanf("%d", &c[j]);
      T[c[j]].parent = l;
      if(j == 0) T[l].left = c[j];
      else T[c[j-1]].right = c[j];
    }
  }

  for(i = 0; i < n; i++) T[i].depth = getDepth(i);

  print(n);
  
  return 0;
}

void print(int n){
  int m, i, j;
  for(i = 0; i < n; i++){
    printf("node %d: parent = %d, depth = %d, ", i, T[i].parent, T[i].depth);
    printf(T[i].parent == -1 ? "root, [" : T[i].left == -1 ? "leaf, [" : "internal node, [");

    m = T[i].left;
    while(1){
      if(m == -1) break;
      printf("%d", m);
      m = T[m].right;
      if(m != -1) printf(", ");
    }
    printf("]\n");
  }
}

int getDepth(int u){
  if(T[u].parent == -1) return 0;
  return getDepth(T[u].parent) + 1;
}

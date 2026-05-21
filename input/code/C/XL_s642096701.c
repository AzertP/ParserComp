#include <stdio.h>
#include <stdlib.h>
#define int long long
#define heap_valtype tuple

typedef struct {
	int A;
	int B;
	int C;
}tuple;

int compare(heap_valtype a, heap_valtype b){
	return a.C - b.C;
}

typedef struct heap_node_sub{
	heap_valtype val;
	struct heap_node_sub *parent;
	struct heap_node_sub *prev;
	struct heap_node_sub *next;
	struct heap_node_sub *left;
	struct heap_node_sub *right;
}heap_node;

typedef struct {
	int N;
	heap_node *root;
	heap_node *last;
}heap;

//heapを生成
heap *make_heap(){
	heap *h = (heap *)malloc(sizeof(heap));
	h->N = 0;
	h->root = NULL;
	h->last = NULL;
	return h;
}

int element_num_heap(heap *h){
	return h->N;
}

void up_heapify(heap_node *n, heap *h){
	if(n->parent != NULL){
		if(compare(n->val, n->parent->val) < 0){
			heap_valtype tmp;
			tmp = n->val;
			n->val = n->parent->val;
			n->parent->val = tmp;
			up_heapify(n->parent, h);
		}
	}
}

//データを追加
void add_data_heap(heap_valtype val, heap *h){
	heap_node *new = (heap_node *)malloc(sizeof(heap_node));
	new->val = val;
	new->next = NULL;
	new->left = NULL;
	new->right = NULL;
	if(element_num_heap(h) == 0){
		new->parent = NULL;
		new->prev = NULL;
		h->root = new;
		h->last = new;
	}
	else if(element_num_heap(h) == 1){
		new->parent = h->root;
		new->prev = h->root;
		h->root->left = new;
		h->root->next = new;
		h->last = new;
		up_heapify(new, h);
	}
	else{
		new->prev = h->last;
		h->last->next = new;
		if(h->last->parent->right == NULL){
			new->parent = h->last->parent;
			h->last->parent->right = new;
		}
		else{
			new->parent = h->last->parent->next;
			h->last->parent->next->left = new;
		}
		h->last = new;
		up_heapify(new, h);
	}
	h->N++;
}

void down_heappify(heap_node *n, heap *h){
	if(n->left != NULL){
		heap_valtype tmp;
		if(n->right == NULL){
			if(compare(n->val, n->left->val) > 0){
				tmp = n->val;
				n->val = n->left->val;
				n->left->val = tmp;
				down_heappify(n->left, h);
			}
		}
		else{
			if(compare(n->left->val, n->right->val) <= 0 && compare(n->val, n->left->val) > 0){
				tmp = n->val;
				n->val = n->left->val;
				n->left->val = tmp;
				down_heappify(n->left, h);
			}
			else if(compare(n->left->val, n->right->val) > 0 && compare(n->val, n->right->val) > 0){
				tmp = n->val;
				n->val = n->right->val;
				n->right->val = tmp;
				down_heappify(n->right, h);
			}
		}
	}
}

//最小のものを取り出す
heap_valtype take_min_heap(heap *h){
	if(element_num_heap(h) == 0){
		printf("no data in the heap\n");
	}
	heap_valtype ans = h->root->val;
	heap_node *ln = h->last;
	if(element_num_heap(h) == 1){
		h->root = NULL;
		h->last = NULL;
	}
	else{
		ln->prev->next = NULL;
		if(ln->parent->right == NULL){
			ln->parent->left = NULL;
		}
		else{
			ln->parent->right = NULL;
		}
		h->last = ln->prev;
		h->root->val = ln->val;
		down_heappify(h->root, h);
	}
	free(ln);
	h->N--;
	return ans;
}

heap_valtype look_min_heap(heap *h){
	if(element_num_heap(h) == 0){
		printf("no data in the heap\n");
	}
	return h->root->val;
}

typedef struct {
	int N;
	int *u;
	int *u_rank;
}union_find;

union_find *make_union_find(int N){
	int i;
	union_find *uf = (union_find *)malloc(sizeof(union_find));
	uf->N = N;
	uf->u = (int *)malloc(sizeof(int) * N);
	uf->u_rank = (int *)malloc(sizeof(int) * N);
	for(i = 0; i < N; i++){
		uf->u[i] = i;
		uf->u_rank[i] = 1;
	}
	return uf;
}

int root_uf(int x, union_find *uf){
	int *u = uf->u;
	if(u[x] == x){
		return x;
	}
	else{
		u[x] = root_uf(u[x], uf);
		return u[x];
	}
}

void combine_uf(int x, int y, union_find *uf){
	int x_root = root_uf(x, uf);
	int y_root = root_uf(y, uf);
	int *u = uf->u;
	int *u_rank = uf->u_rank;
	if(x_root == y_root){
		return;
	}
	else if(u_rank[x_root] < u_rank[y_root]){
		u[x_root] = y_root;
		u_rank[y_root] += u_rank[x_root];
		u_rank[x_root] = 0;
	}
	else{
		u[y_root] = x_root;
		u_rank[x_root] += u_rank[y_root];
		u_rank[y_root] = 0;
	}
}

//xとyが同じ集合に属していれば1を,そうでなければ0を返す
int is_same_union_uf(int x, int y, union_find *uf){
	if(root_uf(x, uf) == root_uf(y, uf)){
		return 1;
	}
	else{
		return 0;
	}
}

//xが属する集合の要素数を返す
int rank_uf(int x, union_find *uf){
	return uf->u_rank[root_uf(x, uf)];
}

signed main(){
	int N, Q, A, B, C, i;
	scanf("%lld%lld", &N, &Q);
	heap *h = make_heap();
	for(i = 0; i < Q; i++){
		scanf("%lld%lld%lld", &A, &B, &C);
		add_data_heap((tuple){A, B, C}, h);
		add_data_heap((tuple){B, (A + 1) % N, C + 1}, h);
	}
	union_find *uf = make_union_find(N);
	tuple t;
	int ans = 0;
	while(element_num_heap(h) > 0){
		t = take_min_heap(h);
		if(is_same_union_uf(t.A, t.B, uf) == 0){
			combine_uf(t.A, t.B, uf);
			ans += t.C;
			add_data_heap((tuple){(t.A + 1) % N, (t.B + 1) % N, t.C + 2}, h);
		}
	}
	printf("%lld\n", ans);
	return 0;
}

typedef struct List {
	struct List *next;
	int v;
} list;

int main()
{
	int i, u, w, z, N, M;
	list **adj = (list**)malloc(sizeof(list*) * (N + 1)), *d = (list*)malloc(sizeof(list) * M * 2);
    int k, flag[100001] = {0}, q[100001], head, tail;
    list *p;
	scanf("%d %d", &N, &M);
	for (i = 1; i <= N; i++) adj[i] = NULL;
	for (i = 0; i < M; i++) {
		scanf("%d %d %d", &u, &w, &z);
		d[i*2].v = w;
		d[i*2+1].v = u;
		d[i*2].next = adj[u];
		d[i*2+1].next = adj[w];
		adj[u] = &(d[i*2]);
		adj[w] = &(d[i*2+1]);
	}
	
	for (i = 1, k = 0; i <= N; i++) {
		if (flag[i] != 0) continue;
		flag[i] = ++k;
		q[0] = i;
		for (head = 0, tail = 1; head < tail; head++) {
			for (p = adj[q[head]]; p != NULL; p = p->next) {
				if (flag[p->v] == 0) {
					flag[p->v] = k;
					q[tail++] = p->v;
				}
			}
		}
	}
	
	printf("%d\n", k);
	fflush(stdout);
	return 0;
}

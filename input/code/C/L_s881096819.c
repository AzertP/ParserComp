#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<inttypes.h>

typedef struct binaryHeap{
  void *array;
  size_t heap_size;
  size_t max_size;
  size_t val_size;
  int (*cmp) (const void *, const void *);
} heap;

heap* new_binary_heap (const size_t val_size, int (*cmpF) (const void *, const void *)) {
  heap *h = (heap *) calloc (1, sizeof (heap));
  h->array = malloc (val_size * (1 + 1));
  h->heap_size = 0;
  h->max_size = 1;
  h->val_size = val_size;
  h->cmp = cmpF;
  return h;
}

int32_t get_heap_size (const heap *h) {
  return h->heap_size;
}

int is_empty (const heap *h) {
  return h->heap_size == 0;
}

void free_heap (heap *h) {
  free (h->array);
  free (h);
}

void init_heap (heap *h) {
  h->heap_size = 0;
}

static inline void heap_func_swap (void * restrict a, void * restrict b, size_t val_size) {
  if ((val_size & 7) == 0) {
    uint64_t *p = (uint64_t *) a;
    uint64_t *q = (uint64_t *) b;
    val_size /= sizeof (uint64_t);
    while (val_size--) {
      const uint64_t tmp = *p;
      *p++ = *q;
      *q++ = tmp;
    }
  } else {
    uint8_t *p = (uint8_t *) a;
    uint8_t *q = (uint8_t *) b;
    while (val_size--) {
      const uint8_t tmp = *p;
      *p++ = *q;
      *q++ = tmp;
    }
  }
}

static inline void heap_func_copy (void * restrict dst, const void * restrict src, size_t val_size) {
  if ((val_size & 7) == 0) {
    const uint64_t *p = (const uint64_t *) src;
    uint64_t *q = (uint64_t *) dst;
    val_size /= sizeof (uint64_t);
    while (val_size--) {
      *q++ = *p++;
    }
  } else {
    const uint8_t *p = (const uint8_t *) src;
    uint8_t *q = (uint8_t *) dst;
    while (val_size--) {
      *q++ = *p++;
    }
  }
}

void push (heap * const h, const void *val) {
  if (h->heap_size == h->max_size) {
    h->max_size = 2 * h->max_size + 1;
    h->array = realloc (h->array, h->val_size * (h->max_size + 1));
  }
  h->heap_size++;
  uint8_t * const array = (uint8_t *) h->array;
  size_t k = h->heap_size;
  const size_t val_size = h->val_size;
  int (* const cmp) (const void *, const void *) = h->cmp;
  heap_func_copy(array + k * val_size, val, val_size);
  while (k > 1) {
    size_t parent = k / 2;
    if (cmp (array + parent * val_size, array + k * val_size) <= 0) {
      return;
    }
    heap_func_swap (array + parent * val_size, array + k * val_size, val_size);
    k = parent;
  }
}

void pop (heap * const h, void *res) {
  uint8_t * const array = (uint8_t *) h->array;
  const size_t val_size = h->val_size;
  if (res != NULL) {
    heap_func_copy (res, array + val_size, val_size);
  }
  heap_func_copy (array + val_size, array + val_size * h->heap_size, val_size);
  h->heap_size--;
  int (* const cmp) (const void *, const void *) = h->cmp;
  const size_t n = h->heap_size;
  size_t k = 1;
  while (2 * k  + 1 <= n) {
    const int c = cmp (array + val_size * 2 * k, array + val_size * (2 * k + 1));
    const size_t next = 2 * k + (c <= 0 ? 0 : 1);
    if (cmp (array + val_size * k, array + val_size * next) <= 0) return;
    heap_func_swap (array + val_size * k, array + val_size * next, val_size);
    k = next;
  }
  if (2 * k <= n && cmp (array + val_size * k, array + val_size * 2 * k) > 0) {
    heap_func_swap (array + val_size * k, array + val_size * 2 * k, val_size);
  }
}

void top (const heap *h, void *res) {
  uint8_t *array = (uint8_t *) h->array;
  const size_t val_size = h->val_size;
  if (res != NULL) {
    heap_func_copy (res, array + val_size, val_size);
  }
}

typedef int32_t i32;
typedef int64_t i64;

static void print_int(i64 n){if(n<0){putchar('-');n=-n;}if(n==0){putchar('0');return;}int s[20],len=0;while(n>0){s[len++]=n%10+'0';n/=10;}while(len>0){putchar(s[--len]);}}
static i64 read_int(void){int prev='\0';int c=getchar();while(!('0'<=c && c<='9')){prev=c;c=getchar();}i64 res=0;while('0'<=c && c<='9'){res=10*res+c-'0';c=getchar();}return prev=='-'?-res:res;}

typedef struct node {
  i32 t;
  i32 x;
} node;

int cmp_node_time (const void *a, const void *b) {
  i32 d = ((node *)a)->t - ((node *)b)->t;
  return d == 0 ? 0 : d < 0 ? -1 : 1;
}

int cmp_int (const void *a, const void *b) {
  i32 d = *(i32 *)a - *(i32 *)b;
  return d == 0 ? 0 : d < 0 ? -1 : 1;
}

void run (void) {
  i32 n = read_int();
  i32 q = read_int();
  node *s = (node *) calloc (n, sizeof (node));
  node *t = (node *) calloc (n, sizeof (node));
  for (i32 i = 0; i < n; ++i) {
    i32 a = read_int();
    i32 b = read_int();
    i32 x = read_int();
    s[i] = (node) {a - x, x};
    t[i] = (node) {b - x, x};
  }
  qsort (s, n, sizeof (node), cmp_node_time);
  qsort (t, n, sizeof (node), cmp_node_time);
  heap *now = new_binary_heap (sizeof (i32), cmp_int);
  heap *del = new_binary_heap (sizeof (i32), cmp_int);
  i32 i = 0, j = 0;
  while (q--) {
    i32 d = read_int();;
    for(; i < n && s[i].t <= d; ++i) {
      push (now, &s[i].x);
    }
    for (; j < n && t[j].t <= d; ++j) {
      push (del, &t[j].x);
    }
    while (!is_empty (now) && !is_empty (del)) {
      i32 a, b;
      top (now, &a);
      top (del, &b);
      if (a != b) break;
      pop (now, NULL);
      pop (del, NULL);
    }
    if (is_empty (now)) {
      puts ("-1");
    } else {
      i32 v;
      top (now, &v);
      print_int (v);
      putchar ('\n');
    }
  }
}

int main (void) {
  run();
  return 0;
}

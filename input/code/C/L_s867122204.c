#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h> // uint64_t

#define BUF_SIZE 40

// size: specify sizeof(str)
int get_str(char *str, int size) {
    if(!fgets(str, size, stdin)) return -1;
    return 0;
}

int get_int(void) {
  int num;
#ifdef BUF_SIZE
  char line[BUF_SIZE];
  if(!fgets(line, BUF_SIZE, stdin)) return 0;
  sscanf(line, "%d", &num);
#else
#error
#endif
  return num;
}

int get_int2(int *a1, int *a2) {
#ifdef BUF_SIZE
  char line[BUF_SIZE];
  if(!fgets(line, BUF_SIZE, stdin)) return -1;
  sscanf(line, "%d %d", a1, a2);
#else
#error
#endif
  return 0;
}

int get_int3(int *a1, int *a2, int *a3) {
#ifdef BUF_SIZE
  char line[BUF_SIZE];
  if(!fgets(line, BUF_SIZE, stdin)) return -1;
  sscanf(line, "%d %d %d", a1, a2, a3);
#else
#error
#endif
  return 0;
}


#define DIVISOR 1000000007
#define AB_MAX 200000

uint64_t get_power2(int n) {
    uint64_t ans = 1;
    uint64_t power2[32];
    power2[0] = 2;
    int i;
    for(i = 1; i < 31; i++) {
        power2[i] = (power2[i-1]*power2[i-1])%DIVISOR;
    }
    for(i = 0; i < 31; i++) {
        if(!(n & (1<<i))) continue;
        ans = (ans * power2[i])%DIVISOR;
    }
    return ans;
}

uint64_t inv_factorial[AB_MAX];

struct pair {
    int x;
    int y;
};

struct pair gcdext(int a, int b) {
    if(a==0) { return (struct pair){0, 1}; }
    struct pair p = gcdext(b%a, a);
    return (struct pair){p.y - b/a * p.x, p.x};
}

void prep_inv(int n) {
    inv_factorial[0] = inv_factorial[1] = 1;
    int i;
    for(i = 2; i <= n; i++) {
        struct pair p = gcdext(i, DIVISOR);
        int inv = (p.x + DIVISOR)%DIVISOR;
        inv_factorial[i] = (inv_factorial[i-1]*inv)%DIVISOR;
    }
}

uint64_t get_combi(int n, int a) {
    uint64_t ans = 1;
    uint64_t i;
    // calc numerator
    for(i = n-a+1; i <= n; i++) {
        ans = (ans * i)%DIVISOR;
    }
    ans = (ans * inv_factorial[a])%DIVISOR;
#ifdef DEBUG
    printf("get_combi: %llu\n", ans);
#endif
    return ans;
}

int main(void) {
    int n, a, b;
    get_int3(&n, &a, &b);
    uint64_t ans = get_power2(n) - 1; // n_C_0 = 1
#ifdef DEBUG
    printf("%llu\n", ans);
#endif
    prep_inv(b);
    ans = (ans + DIVISOR - get_combi(n, a))%DIVISOR;
    ans = (ans + DIVISOR - get_combi(n, b))%DIVISOR;
    printf("%d\n", (int)ans);
    return 0;
}
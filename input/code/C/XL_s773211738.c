


long long fac[MAX], finv[MAX], inv[MAX];

//prototype declaration
int i_in();
long long ll_in();
double d_in();
char c_in();
void s_in(char *s);
void i_out(int i);
void ll_out(long long i);
void d_out(double i);
void c_out(char c);
void s_out(char *s);
int gcd(int a, int b);
int lcm(int a, int b);
long long ll_gcd(long long a, long long b);
long long ll_lcm(long long a, long long b);
void sort_asc(int *a);
void sort_desc(int *a);
int asc(const void *a, const void *b);
int desc(const void *a, const void *b);
void ll_sort_asc(long long *a);
void ll_sort_desc(long long *a);
int ll_asc(const void *a, const void *b);
int ll_desc(const void *a, const void *b);
double to_deg(double rad);
double distance(int xi, int yi, int xj, int yj);
void initComb();
long long comb(int n, int k);

//main function
int main() {
    long long x = ll_in(),c = 0;
    for (long long i = 100; i < x; i *= 1.01) {
        c++;
    }
    ll_out(c);
    return 0;
}

//body function
int i_in() {
    int i;
    scanf("%d", &i);
    return i;
}

long long ll_in() {
    long long i;
    scanf("%lld", &i);
    return i;
}

double d_in() {
    double i;
    scanf("%lf", &i);
    return i;
}

char c_in() {
    char c;
    scanf("%c", &c);
    return c;
}

void s_in(char *s) {
    scanf("%s", s);
}

void i_out(int i) {
    printf("%d\n", i);
}

void ll_out(long long i) {
    printf("%lld\n", i);
}

void d_out(double i) {
    printf("%lf\n", i);
}

void c_out(char c) {
    printf("%c\n", c);
}

void s_out(char *s) {
    printf("%s\n", s);
}

/*
    int
*/

int gcd(int a, int b) {
    int r;

    if (a < b) {
        r = a;
        a = b;
        b = r;
    }

    r = a % b;
    while (r != 0) {
        a = b;
        b = r;
        r = a % b;
    }

    return b;
}

/*
    int
*/

int lcm(int a, int b) {
    int r = ll_gcd(a, b);
    a /= r;
    return a * b;
}

/*
    long long
*/

long long ll_gcd(long long a, long long b) {
    long long r;

    if (a < b) {
        r = a;
        a = b;
        b = r;
    }

    r = a % b;
    while (r != 0) {
        a = b;
        b = r;
        r = a % b;
    }

    return b;
}

/*
    long long
*/

long long ll_lcm(long long a, long long b) {
    long long r = ll_gcd(a, b);
    a /= r;
    return a * b;
}

/*
    int
*/

void sort_asc(int *a) {
    qsort(a, sizeof(a)/sizeof(int), sizeof(int), asc);
}

/*
    int
*/

void sort_desc(int *a) {
    qsort(a, sizeof(a)/sizeof(int), sizeof(int), desc);
}

/*
    int
*/

int asc(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

/*
    int
*/

int desc(const void *a, const void *b) {
    return *(int *)b - *(int *)a;
}

/*
    long long
*/

void ll_sort_asc(long long *a) {
    qsort(a, sizeof(a)/sizeof(long long), sizeof(long long), ll_asc);
}

/*
    long long
*/

void ll_sort_desc(long long *a) {
    qsort(a, sizeof(a)/sizeof(long long), sizeof(long long), ll_desc);
}

/*
    long long
*/

int ll_asc(const void *a, const void *b) {
    long long *A = (long long *)a;
    long long *B = (long long *)b;
    if (*A > *B) return 1;
    else if (*A < *B) return -1;
    else return 0;
}

/*
    long long
*/

int ll_desc(const void *a, const void *b) {
    long long *A = (long long *)a;
    long long *B = (long long *)b;
    if (*A < *B) return 1;
    else if (*A > *B) return -1;
    else return 0;
}

/*
    
*/

double to_deg(double rad) {
    return rad * 180.0 / (atan(1.0) * 4.0);
}

/*
    
*/

double distance(int xi, int yi, int xj, int yj) {
    return hypot(xj - xi, yj - yi);
}

/*
    
*/

void initComb() {
    fac[0] = fac[1] = 1;
    finv[0] = finv[1] = 1;
    inv[1] = 1;
    for (int i = 2; i < MAX; i++){
        fac[i] = fac[i - 1] * i % MOD;
        inv[i] = MOD - inv[MOD%i] * (MOD / i) % MOD;
        finv[i] = finv[i - 1] * inv[i] % MOD;
    }
}

/*
    nCk
*/

long long comb(int n, int k){
    if (n < k) return 0;
    if (n < 0 || k < 0) return 0;
    return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD;
}

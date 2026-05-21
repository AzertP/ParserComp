
#pragma warning(disable:4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#pragma region 定義
#define i_cin(X) scanf("%d", &X)
#define i_cin2(X,Y) scanf("%d %d",&X,&Y)
#define i_cin4(X,Y,Z,W) scanf("%d %d %d %d",&X,&Y,&Z,&W)
#define i_cin3(X,Y,Z) scanf("%d %d %d",&X,&Y,&Z)
#define l_cin(X) scanf("%ld",&X)
#define f_cin(X) scanf("%f",&X)
#define l_cin2(X,Y) scanf("%ld %ld",&X,&Y)
#define s_cin(X) scanf("%s",&X)
#define c_cin(X) scanf("%c",&X);
#define ll_cin(X) scanf("%lld", &X)
#define ull_cin(X) scanf("%llu", &X)
#define lf_cin(X) scanf("%lf", &X)

#define  i_cout(X) printf("%d\n",X)
#define  f_cout(X) printf("%g\n",X)
#define  i_cout2(X,Y) printf("%d %d\n",X,Y)
#define  l_cout(X) printf("%ld\n",X)
#define  s_cout(X) printf("%s\n",X)
#define  s_coutc(X) printf("%s",X)

#define  c_cout(X) printf("%c",X)


#define  ll_cout(X) printf("%lld\n",X)
#define  ull_cout(X) printf("%llu\n",X)



typedef long long  ll;
typedef unsigned long long  ull;

#define rept(x, s, n) for (int x = s; x < n; x++)

#define S_RTN(S)  s_cout(S);return 0


#define _itoa(A,N) sprintf(A, "%d", N);

int i_cins(int n, int* A);
int l_cins2(int n, long* A, long* B);
int s_dsort(const void* a, const void* b);
int s_asort(const void* a, const void* b);

int  _gcd(int a, int b);
int  _swp(int* a, int* b);
int _cknum(char* a, int n);
int _atoi(char* s, int len);


int s_asorts(const void* a, const void* b);

//昇順
int s_asorts(const void* a, const void* b) {
	return(strcmp((char*)a, (char*)b));
}
int s_dsort(const void* a, const void* b) {
	return(*(int*)b - *(int*)a);
}
int s_asort(const void* a, const void* b) {
	return(*(int*)a - *(int*)b);
}
int l_cins2(int n, long* a, long* b) {
	int i;
	rept(i, 0, n) {
		l_cin2(*(a + i), *(b + i));
	}
	return 0;
}
int i_cins(int n, int* a) {
	int i;
	for (i = 0; i < n; i++) {
		i_cin(*(a + i));
	}

	return 0;
}
int _gcd(int a, int b) {
	int r, tmp;
	if (a < b) { tmp = a; a = b; b = tmp; }
	r = a % b;
	while (r) {
		if (!r) break;
		a = b; b = r;
		r = a % b;
	}
	return(b);
}

//#define pai  3.1415926535      
#define deg_rad(X)  (X*(M_PI/180))

long _max(long a, long b) {
	return a > b ? a : b;
}
long _min(long a, long b) {
	return a < b ? a : b;
}
int _swp(int* a, int* b)
{
	int tmp;
	tmp = *b; *b = *a; *a = tmp;
	return 0;
}

int _cknum(char* a, int n) {
	int i;
	char t = '0';

	for (i = 0; i < n; i++) {
		if (a[i] < '0' || a[i]>'9') return 1;
	}
	return 0;
}


int _atoi(char* s, int len) {
	char tmp[20];
	memcpy(tmp, s, len);
	tmp[len] = 0x00;
	return (atoi(tmp));
}

//素因数分解　pf素数が入る　pc 素数の数
ll _pfact(ll a);
static ll pf[100000000];
static ll pc;

ll _pfact(ll a) {
	long i, r;
	if (a % 2 == 0) {
		pf[pc] = 2; pc++;
		return(_pfact(a / 2));
	}
	r = sqrt(a);
	for (i = 3; i <= r; i += 2) {
		if (a % i == 0) {
			pf[pc] = i; pc++;
			return(_pfact(a / i));
		}
	}
	if (a != 1) {
		pf[pc] = a; pc++;
	}
	return 0;

}


#pragma endregion


//144-C

int main(void) {
	ll n;
	ll tmp, r;

	ll_cin(n);
	tmp = sqrt(n);
	if (tmp * tmp == n)
	{
		ll_cout(tmp * 2 - 2); return 0;
	}
	ll i;

	for (i = tmp; i > 1; i--) {
		if (n % i) continue;
		break;
	}
	if (i == 1) {
		ll_cout(n-1); return 0;
	}
	ll ans;

	ans= i + n / i - 2;
	ll_cout(ans);
	return 0;

	/*r = _pfact(n);
	if (pc == 1) {
		ll_cout(tmp); return 0;
	}

	ll_cout(pc);
	for (int i = pc-1; i >= 0; i--)
	     ll_cout(pf[i]);*/
}


//157-C
//int main(void) {
//	int n, m;
//	int s, c;
//	char a[4] = { 0,0,0,0 };
//	int i, ans=0;
//	
//	i_cin2(n, m);
//
//	for (i = 0; i < m; i++) {
//		i_cin2(s, c);
//		if (n > 1 && s == 1 && !c) continue;
//		if(a[s-1])
//		  a[s - 1] = _min(a[s - 1], c + '0');
//		else 
//		  a[s - 1] = c + '0';
//	}
//	for (i = 0; i < n; i++)
//		if (!a[i])a[i] ='0';
//
//	a[n] = 0;
//	ans = atoi(a);
//	if (n == 3 && (ans < 10 || ans < 100)) ans = -1;
//	if (n == 2 && ans < 10) ans = -1;
//	i_cout(ans);
//	return 0;
//}

	
 //ABC-07 D 幅優先探索
//#define Q_SIZE 100000
//static struct _queue {
//	char *d[Q_SIZE];
//	int h;
//	int t;
//} q;
//
//int  pushq(struct _queue* q, char *item);
//int  pushq(struct _queue* q, char *item) {
//	if (q->t == Q_SIZE) return -1;
//	q->d[q->t] = item;
//	q->t++;
//	return 0;
//}
//
//char* popq(struct _queue* q);
//char* popq(struct _queue* q) {
//	char* tmp;
//	if (q->h == q->t) return -1;
//	tmp = q->d[q->h];
//	q->h--;
//	return tmp;
//}
//
//static struct _p {
//	int x;
//	int y;
//	int cnt;
//} q;
//static char a[51][51];
//static struct _p p[50];
//static int    pcnt;
//
//int _enq(struct _p *p, int x, int y, int cnt);
//int _enq(struct _p *p,int x, int y, int cnt) {
//	p->x = x; p->y = y; p->cnt = cnt;
//	pushq(&q,p); pcnt++;
//}
//int _deq(struct _p* p);
//int _deq(struct _p* p) {
//	p=popq(&q);
//	return(p == -1 ? p : 0);
//}
//int sqr_chk(int x, int y, int cnt) {
//	int tmp;
//	tmp = pcnt;
//	if (a[x - 1][y - 1] == '.') 
//	{
//		_enq(&p[pcnt], x - 1, y - 1, cnt); a[x - 1][y - 1] = 'X';
//	}
//	if (a[x - 1][y] == '.')
//	{
//		_enq(&p[pcnt], x - 1, y, cnt); a[x - 1][y] = 'X';
//	}
//	if (a[x+1][y] == '.') 
//	{
//		_enq(&p[pcnt], x + 1, y, cnt); a[x + 1][y] = 'X';
//	}
//	if (a[x][y+1] == '.') 
//	{
//		_enq(&p[pcnt], x, y + 1, cnt); a[x][y + 1] = 'X';
//	}
//	if (pcnt == tmp) return -1;
//	else             return 0;
//}
//
//int main(void) {
//	int r, c;
//	int sy, sx, gy, gx;
//	int i = 0, j = 0, x = 0, y = 0, ans = 0;
//	char cnt = '0';
//
//	i_cin(r, c);
//	i_cin4(sx, sy, gx, gy);
//	if (sx == gx && sy == gy) {
//		i_cout(0); return 0;
//	}
//	for (i = 0; i < r; i++)
//		s_cin(a[i]);
//	sy--; sx--;
//	struct _p* pt;
//	pt = &p[cnt];
//	p->x = sx; p->y = sy; p->cnt = cnt;
//
//	int rc;
//	while (1) {
//		rc = sqr_chk(pt->x,pt->y,pt->cnt);
//		if (rc == -1) {
//			rc = _deq(pt);
//			if (rc == -1 || (pt->x == gx && pt->y == gy)) break;
//		}
//	}
//
//	if (rc == -1) i_cout(rc);
//	else i_cout(ans);
//	return 0;
//}
	
	
	



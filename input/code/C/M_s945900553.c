#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define ll long long
#define rep(i,n) for(ll i=0;i<(n);i++)
#define max(p,q) ((p)>(q)?(p):(q))
#define min(p,q) ((p)<(q)?(p):(q))
#define chmax(a,b) ((a)=(a)>(b)?(a):(b))
#define chmin(a,b) ((a)=(a)<(b)?(a):(b))
#define abs(p) ((p)>=(0)?(p):(-(p)))
#define swap(a,b) do{ll w=a;a=b;b=w;}while(0)
#define swapd(a,b) do{double w=a;a=b;b=w}while(0)
#define in(a) scanf("%lld", &(a))
#define ind(a) scanf("%lf", &(a))
#define put(a) printf("%lld\n", (a))
#define putd(a) printf("%.15f\n", (a));
//def   puts(a) printf("%s\n", a) 文字はこっち
int cmp(const void *a, const void *b);
//your code here!

int main(void){
    ll p=0, idx[3]={0};
    char s[3][101];
    rep(i,3) scanf("%s", s[i]);
    while(1){
        if(s[p][idx[p]]=='\0'){
            putchar('A'+p);
            return 0;
        }
        if(s[p][idx[p]]=='a'){
            idx[p]++;
            p=0;
        }
        else if(s[p][idx[p]]=='b'){
            idx[p]++;
            p=1;
        }
        else {
            idx[p]++;
            p=2;
        }
    }
}

int cmp(const void *a, const void *b){
    ll A=*(ll *)a, B=*(ll *)b;
    if(A==B)return 0;
    else return A>B ? 1:-1;//昇順ソート 1,2....9
}

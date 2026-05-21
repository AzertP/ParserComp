#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define P 1000000007

int comp(const void *a, const void *b){return *(int*)a-*(int*)b;}

char s[100001];
int a[100000];
int main(void){
	scanf("%s", s);
	int n=strlen(s), i;

	int f, g;
	f=0;
	for(i=0; i<n; i++){
		if(s[i]=='L'){
			if(f==0){
				f=1;
				g=i;
				a[g]=1;
			}else{
				a[g]++;
			}
		}else{
			f=0;
		}
	}
	f=0;
	for(i=n-1; i>=0; i--){
		if(s[i]=='R'){
			if(f==0){
				f=1;
				g=i;
				a[g]=1;
			}else{
				a[g]++;
			}
		}else{
			f=0;
		}
	}
	for(i=0; i<n; i++){
		if(a[i]!=0){
			int R=a[i];
			int L=a[i+1];
			a[i]=(R+1)/2+L/2;
			a[i+1]=(L+1)/2+R/2;
			i++;
		}
	}
	for(i=0; i<n; i++){
		printf("%d ", a[i]);
	}
	putchar(10);
}

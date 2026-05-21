#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int sort(const void *a, const void *b){
    return *(int*)a-*(int*)b ;
    }
    
int main(void){
    int a[3],ans=0;
    scanf("%d%d%d",&a[0],&a[1],&a[2]);
    qsort (a,3,sizeof(int),sort);   
    ans+=a[2]-a[1];
    if ((a[1]-a[0])%2==0){
        ans+=(a[1]-a[0])/2 ;
    } else {
        ans+=((a[1]-a[0]+1)/2)+1;
    }
    printf ("%d\n",ans);
    return 0 ;
}
#include<stdio.h>
int main(void){
    int n;
    int a[101];
    int i,j;
    int temp;
    scanf("%d",&n);
    int k=n-1;
    for(i=0;i<n;i++)
        scanf("%d",&a[i]);
    for(j=0;j<n/2;j++){
        temp=a[j];
        a[j]=a[k];
        a[k]=temp;
        k--;
    }
    for(i=0;i<n;i++){
        printf("%d",a[i]);
        if(i!=n-1)
            printf(" ");
    }
    printf("\n");
    return 0;
}
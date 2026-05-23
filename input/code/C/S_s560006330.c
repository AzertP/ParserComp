
int main(void){
    int n,i,x,max,min;
    long long sum=0;
    scanf("%d",&n);
    scanf("%d",&x);
    min=x;
    max=x;
    sum+=x;
    for(i=1;i<n;i++){
        scanf("%d",&x);
        if(x<min) min=x;
        if(x>max) max=x;
        sum+=x;
    }
    
    printf("%d %d %ld\n",min,max,sum);
    
    return 0;
}

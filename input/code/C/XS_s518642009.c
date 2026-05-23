int main(void){
    int n,ans=1;
    scanf("%d",&n);
    while(ans+ans<=n){
        ans+=ans;
    };
    printf("%d",ans);
    return 0;
};

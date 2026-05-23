int main(){
int n,ans=0,cnt;
scanf("%d",&n);
for(int i=1;i<=n;i+=2){
cnt=0;
for(int j=1;j<=i;j++)if(i%j==0)cnt++;
if(cnt==8)ans++;
}
printf("%d\n",ans);
}

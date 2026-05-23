

int main(){
	int n,k,i,x,ans=0;
	scanf("%d%d",&n,&k);
	for(i=0;i<n;i++){
		scanf("%d",&x);
		ans+=min(abs(x-k),abs(x))*2;
	}
	printf("%d\n",ans);
	return 0;
}

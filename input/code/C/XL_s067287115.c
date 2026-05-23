



static llong int heap[100010];
int sz=-1;

void push(llong int x){
	//
	int i;
	sz++;
	i=sz;
	
	while(i>0){
		//
		int p=(i-1)/2;
		//
		if(heap[p]>=x) break;
		//
		heap[i]=heap[p];
		i=p;
	}
	heap[i]=x;
}

llong int pop(){
	llong int ret;
	int a,b;
	llong int x;
	int i;
	//
	ret=heap[0];
	//
	x=heap[sz];
	sz=sz-1;
	//
	i=0;
	while(i*2+1<=sz){
		a=i*2+1;
		b=i*2+2;
		if(b<sz&&heap[b]>heap[a]) a=b;
		//
		if(heap[a]<=x) break;
		//
		heap[i]=heap[a];
		i=a;
	}
	heap[i]=x;
	
	return ret;
}



static llong int heapup[100010];
int szup=-1;
 
void pushup(llong int x){
	//
	int i;
	int p;
	szup++;
	i=szup;
	
	while(i>0){
		//
		p=(i-1)/2;
		//
		if(heapup[p]>=x) break;
		//
		heapup[i]=heapup[p];
		i=p;
	}
	heapup[i]=x;
}
 
int popup(){
	llong int ret;
	int a,b;
	llong int x;
	int i;
	//
	ret=heapup[0];
	//
	x=heapup[szup];
	szup=szup-1;
	//
	i=0;
	while(i*2+1<=szup){
		a=i*2+1;
		b=i*2+2;
		if(b<=szup&&heapup[b]>heapup[a]) a=b;
		//
		if(heapup[a]<=x) {
			break;
		}
		//
		heapup[i]=heapup[a];
		i=a;
	}
	heapup[i]=x;
	
	return ret;
}
 
 
 int main(void)
{
	//
	int n;
	llong int c;
	static llong int x[100010];
	static llong int v[100010];
	static llong int rsum[100010];
	static llong int lsum[100010];
	static llong int rura[100010];
	static llong int lura[100010];
	static llong int rmax[100010];
	static llong int lmax[100010];
	
	
	//
	int i,j,k,l;
	int flag=0;
	llong int ans=0;
	int count=0;
	llong int temp,temp1,temp2;
	int max,min;
	int len;
	//
	
	scanf("%d %lld",&n,&c);
	
	for(i=0;i<n;i++){
		scanf("%lld %lld",&x[i],&v[i]);
	}
	
//	for(i=0;i<n;i++){
//		printf("%lld %lld",x[i],v[i]);
//	}
	
	
//	printf("n%d\n", n);
//	printf("\n");
	//
	
//	printf("v[0]=%lld,x[0]=%lld\n",v[0],x[0]);
	for(i=0;i<n;i++){
		if(i==0){
			rsum[0]=v[0]-x[0];
		}else{
			rsum[i]=rsum[i-1]+v[i]-(x[i]-x[i-1]);
		}
		if(i==0){
			rura[0]=v[0]-2*x[0];
		}else{
			rura[i]=rura[i-1]+v[i]-2*(x[i]-x[i-1]);
		}
	}
	
//	printf("v[2]=%lld,c=%lld,x[2]=%lld\n",v[2],c,x[2]);
//	printf("c-x[n-1]=%lld\n",c-x[n-1]);
	for(i=n-1;i>=0;i--){
		if(i==n-1){
			lsum[n-1]=v[n-1]-(c-x[n-1]);
		}else{
			lsum[i]=lsum[i+1]+v[i]-((c-x[i])-(c-x[i+1]));
		}
		
		if(i==n-1){
			lura[n-1]=v[n-1]-2*(c-x[n-1]);
		}else{
			lura[i]=lura[i+1]+v[i]-2*((c-x[i])-(c-x[i+1]));
		}
	}
	
	for(i=0;i<n;i++){
		push(rsum[i]);
		rmax[i]=heap[0];
	}
	
	for(i=n-1;i>=0;i--){
		pushup(lsum[i]);
		lmax[i]=heapup[0];
	}
	
/*	for(i=0;i<n;i++){
		printf("rsum[%d]=%lld\n",i,rsum[i]);
	}
	for(i=0;i<n;i++){
		printf("lsum[%d]=%lld\n",i,lsum[i]);
	}
	for(i=0;i<n;i++){
		printf("rura[%d]=%lld\n",i,rura[i]);
	}
	for(i=0;i<n;i++){
		printf("lura[%d]=%lld\n",i,lura[i]);
	}
	for(i=0;i<n;i++){
		printf("rmax[%d]=%lld\n",i,rmax[i]);
	}
	for(i=0;i<n;i++){
		printf("lmax[%d]=%lld\n",i,lmax[i]);
	}
*/	
	ans=max(max(rmax[n-1],lmax[0]),0);
	
	for(i=1;i<n;i++){
		temp=rmax[i-1]+lura[i];
		if(ans<temp) ans=temp;
	}
	
	for(i=0;i<n-1;i++){
		temp=lmax[i+1]+rura[i];
		if(ans<temp) ans=temp;
	}
	
//	printf("\n");
	//
	
	printf("%lld",ans);
	
//	printf("\n");
	
	return 0;
	
}


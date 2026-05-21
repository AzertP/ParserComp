d[100];i,j,k;s;main(h){
	for(gets(d);j<100;)scanf("%d",d+j++);
	for(;i<10;i++)for(j=0;j<10;j++)for(k=0;k<10;k++)d[j*10+k]=fmin(d[j*10+k],d[j*10+i]+d[i*10+k]);
	for(;~scanf("%d",&h);)~h&&(s+=d[h*10+1]);
	printf("%d",s);
}
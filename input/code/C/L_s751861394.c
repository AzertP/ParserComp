 
int main(void){

	int A[3][10],B[3][10],C[3][10],D[3][10],i,ii,n,b[300],f[300],r[300],v[300];

	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			A[i][ii]=0;
		}
	}
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			B[i][ii]=0;
		}
	}
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			C[i][ii]=0;
		}
	}
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			D[i][ii]=0;
		}
	}

	scanf("%d",&n);
	for(i=0;i<n;i++){
		scanf("%d %d %d %d",&b[i],&f[i],&r[i],&v[i]);
	}

	for(i=0;i<n;i++){
		if(b[i]==1){
			A[f[i]-1][r[i]-1]+=v[i];
		}
		if(b[i]==2){
			B[f[i]-1][r[i]-1]+=v[i];
		}
		if(b[i]==3){
			C[f[i]-1][r[i]-1]+=v[i];
		}
		if(b[i]==4){
			D[f[i]-1][r[i]-1]+=v[i];
		}
	}

	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			printf(" %d",A[i][ii]);
		}
		printf("\n");
	}
	printf("####################\n");
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			printf(" %d",B[i][ii]);
		}
		printf("\n");
	}
	printf("####################\n");
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			printf(" %d",C[i][ii]);
		}
		printf("\n");
	}
	printf("####################\n");
	for(i=0;i<3;i++){
		for(ii=0;ii<10;ii++){
			printf(" %d",D[i][ii]);
		}
		printf("\n");
	}

	return 0;

}

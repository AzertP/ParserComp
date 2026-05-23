int main(void)
{
	int n[100],num,i;
	scanf("%d",&num);
	for(i=0;i<num;i++)
		scanf("%d",&n[i]);
	for(i=0;i<num;i++){
		printf("%d",n[num-i-1]);
		if(num-i-1 !=0)
			printf(" ");
		else
			printf("\n");
	}
	return 0;
	
}

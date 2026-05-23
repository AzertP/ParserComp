
int main(){
	char s[100],t[100],sum[300];
	int c=0;
	int n;
	scanf("%d",&n);
	scanf("%s%s",s,t);
	for(int i=0;i<strlen(s);i++){
		printf("%c",s[i]);
		printf("%c",t[i]);
	}

	return 0;
}

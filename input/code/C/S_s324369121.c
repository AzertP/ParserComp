int main()
{
	int i,t,a=753;
	char s[11];

	scanf("%s",s);
	for(i=0;i<strlen(s)-2;i++){
		t=abs((s[i]-'0')*100+(s[i+1]-'0')*10+(s[i+2]-'0')-753);
		if(a>t)
			a=t;
	}
	printf("%d",a);
	return 0;
}

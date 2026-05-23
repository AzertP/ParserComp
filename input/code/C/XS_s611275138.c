
int main()
{
  char s[101];
  gets(s);
  
  int len = strlen(s);
  
  int i,sum = 0;
  
  for(i = 0;i <= (len - 1)/2;i++){if(s[i] != s[len-1-i])sum++;}
  printf("%d",sum);
}

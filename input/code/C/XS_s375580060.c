#include <stdio.h>
#include <math.h>
int main(void){
char s[200005];
int sum=0;
scanf("%s",s);
for(int i=0;s[i]!='\0';i++){
	sum=sum+s[i]-48;
}
if(sum%9==0){printf("Yes");}
else{printf("No");}
}
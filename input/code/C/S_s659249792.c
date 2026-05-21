#include <stdio.h>

int main(void){

	char line[11],line1[30];
	int i=0,sum=0;

	scanf("%s",line);
		while(line[i]){
			if(isupper(line[i]))
				line[i]=tolower(line[i]);
			i++;
		}
		for(;i=0,scanf("%s",line1),strcmp(line1,"END_OF_TEXT");){
			while(line1[i]){
				if(isupper(line1[i]))
					line1[i]=tolower(line1[i]);
				i++;
			}
			if(!strcmp(line1,line))
				sum++;
			}
		printf("%d\n",sum);
	return 0;
}
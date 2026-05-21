
#include <stdio.h>


int main (){

int dataset=0;
int year[100];
int month[100];
int date[100];
int input=0;
int result[100]={};
int dataset_num=0;
int dataset_num2=0;
int count=0;
int buf;


scanf("%d",&dataset);
dataset_num = dataset;
dataset_num2 = dataset;


while( dataset != 0 ){
	scanf("%d %d %d",&year[input],&month[input],&date[input]);
	dataset--;
	input++;
}


while(dataset_num != 0){

	if(year[count]% 3 == 0){

		/*??\?????????????????????*/
		while ( 1 ){
			result[count]++;
			date[count]++;
	
			if(date[count]==21){
				month[count]++;
				break;
			}
		}	

		/*??????????????\?????????????????????*/
		while ( month[count] != 11){
			result[count] += 20;
			month[count]++;
		}	
	}

	else{

		if(month[count] % 2 == 1){
			buf = 21 - date[count];
			result[count] += buf;
				
		}

		else if(month[count] % 2 == 0 ){
			buf = 20 - date[count];
			result[count] += buf;
		}

		month[count]++;
		/*??????????????\?????????????????????*/
		while ( month[count] != 11){

			if(month[count] % 2 == 1){
				result[count] += 20;
			}

			else if(month[count] % 2 == 0 ){
				result[count] += 19;
			}
			month[count]++;
		}
	}

	year[count]++;
	while(1){

		if(year[count]==1000){
			break;
		}
		if(year[count] % 3 == 0){

			result[count] += 200;
		}
		else{
			result[count] += 195;
		}
		year[count]++;

		

	}

	count++;
	dataset_num--;
}

count=0;
while(1){
	printf("%d\n",result[count] );
	count++;

	if(count == dataset_num2){
		break;
	}

}
	

	return 0;
}
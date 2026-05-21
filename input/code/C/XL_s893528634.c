#include<stdio.h>
#include<string.h>

#define U 1	//上
#define D 2	//下
#define S 3 //南 
#define E 4 //東
#define N 5 //北
#define W 6 //西


int d1=U , d2=S , d3=E;		
int sum = 0;


void West(void){

	switch(d1){

	case U:
		d1 = W;
		break;

	case W:
		d1 = D;
		break;

	case D:
		d1 = E;
		break;

	case E:
		d1 = U;
		break;

	}
	
	switch(d2){

	case U:
		d2 = W;
		break;

	case W:
		d2 = D;
		break;

	case D:
		d2 = E;
		break;

	case E:
		d2 = U;
		break;

	}

	
	switch(d3){

	case U:
		d3 = W;
		break;

	case W:
		d3 = D;
		break;

	case D:
		d3 = E;
		break;

	case E:
		d3 = U;
		break;

	}


}
void East(void){

	switch(d1){

	case U:
		d1 = E;
		break;

	case W:
		d1 = U;
		break;

	case D:
		d1 = W;
		break;

	case E:
		d1 = D;
		break;

	}
	
	switch(d2){

	case U:
		d2 = E;
		break;

	case W:
		d2 = U;
		break;

	case D:
		d2 = W;
		break;

	case E:
		d2 = D;
		break;

	}

	
	switch(d3){

	case U:
		d3 = E;
		break;

	case W:
		d3 = U;
		break;

	case D:
		d3 = W;
		break;

	case E:
		d3 = D;
		break;

	}

}
void North(void){

	switch(d1){

	case U:
		d1 = N;
		break;
	case N:
		d1 = D;
		break;
	case D:
		d1 = S;
		break;
	case S:
		d1 = U;
		break;

	}
	switch(d2){

	case U:
		d2 = N;
		break;
	case N:
		d2 = D;
		break;
	case D:
		d2 = S;
		break;
	case S:
		d2 = U;
		break;

	}
	switch(d3){

	case U:
		d3 = N;
		break;
	case N:
		d3 = D;
		break;
	case D:
		d3 = S;
		break;
	case S:
		d3 = U;
		break;

	}


}
void South(void){

		switch(d1){

	case U:
		d1 = S;
		break;
	case N:
		d1 = U;
		break;
	case D:
		d1 = N;
		break;
	case S:
		d1 = D;
		break;

	}
	switch(d2){

	case U:
		d2 = S;
		break;
	case N:
		d2 = U;
		break;
	case D:
		d2 = N;
		break;
	case S:
		d2 = D;
		break;

	}
	switch(d3){

	case U:
		d3 = S;
		break;
	case N:
		d3 = U;
		break;
	case D:
		d3 = N;
		break;
	case S:
		d3 = D;
		break;

	}
}
void Right(void){

	if(d1 == S||d1 == E||d1 == N||d1 == W){
		d1--;
		if(d1 == 2)
			d1 = W;
	}
	if(d2 == S||d2 == E||d2 == N||d2 == W){
		d2--;
		if(d2 == 2)
			d2 = W;
	}
	if(d3 == S||d3 == E||d3 == N||d3 == W){
		d3--;
		if(d3 == 2)
			d3 = W;
	}

}
void Left(void){

	if(d1 == S||d1 == E||d1 == N||d1 == W){
		d1++;
		if(d1 == 7)
			d1 = S;
	}
	if(d2 == S||d2 == E||d2 == N||d2 == W){
		d2++;
		if(d2 == 7)
			d2 = S;
	}
	if(d3 == S||d3 == E||d3 == N||d3 == W){
		d3++;
		if(d3 == 7)
			d3 = S;
	}
}

void Sum(void){

	if(d1 == U)
		sum += 1;
	else if(d1 == D)
		sum += 6;
	else if(d2 == U)
		sum += 2;
	else if(d2 == D)
		sum += 5;
	else if(d3 == U)
		sum += 3;
	else if(d3 == D)
		sum += 4;

}

int main(void){

	int n,i;

	char s[6];


	while(1){

		sum = 0;
		d1=U , d2=S , d3=E;
		
		scanf("%d",&n);
		if(n==0)
			break;

		for(i=0;i<n;i++){

			scanf("%s",s);

			if(strcmp(s,"West")==0){
				West();
			}
			else if(strcmp(s,"East")==0){
				East();
			}
			else if(strcmp(s,"North")==0){
				North();
			}
			else if(strcmp(s,"South")==0){
				South();
			}
			else if(strcmp(s,"Right")==0){
				Right();
			}
			else if(strcmp(s,"Left")==0){
				Left();
			}


			Sum();
		}

		printf("%d\n",sum+1);

	}


	return 0;
}
			
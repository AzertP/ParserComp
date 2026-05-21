#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>



/*以下便利なマクロを定義する。*/

#define rep(i, min, max) for(i=min; i<=max; i=i+1)



#define if_forall(i, min, max, prop)\
\
	rep(i, min, max)\
	{\
		if(!(prop))\
		{\
			break;\
		}\
	}\
\
	if(i==max+1)\



#define if_exists(i, min, max, prop)\
\
	rep(i, min, max)\
	{\
		if(prop)\
		{\
			break;\
		}\
	}\
\
	if(i<max+1)\



#define sum(i, min, max, sequence, answer)\
{\
	answer=0;\
	rep(i, min, max)\
	{\
		answer=answer+sequence;\
	}\
}\



#define prod(i, min, max, sequence, answer)\
{\
	answer=1;\
	rep(i, min, max)\
	{\
		answer=answer*sequence;\
	}\
}\











/*以下便利な関数を定義する。*/

void fill_int(int array[], int min, int max, int element)
{
	int i=min;

	rep(i, min, max)
	{
		array[i]=element;
	}

}



void fill_char(char array[], int min, int max, char element)
{
	int i=min;

	rep(i, min, max)
	{
		array[i]=element;
	}

}



void  swap_int(int* x, int* y)
{
    int  tmp;

    tmp = *x;
    *x = *y;
    *y = tmp;
}



void  swap_int_array(int x[], int y[], int min, int max)
{
	int i;

    rep(i, min, max)
	{
		swap_int(&x[i], &y[i]);
	}
}



void  swap_char(char* x, char* y)
{
    char  tmp;

    tmp = *x;
    *x = *y;
    *y = tmp;
}



void  swap_char_array(char x[], char y[], int min, int max)
{
	int i;

    rep(i, min, max)
	{
		swap_char(&x[i], &y[i]);
	}
}



void reverse_char(char array[], int min, int max)
{
	int i;

	rep(i, min, (min+max)/2)
	{
		swap_char(&array[i], &array[min+max-i]);
	}
}



int max_int(int var1, int var2)
{
	int answer=var1;

	if(answer<var2)
	{
		answer=var2;
	}

	return answer;
}



int min_int(int var1, int var2)
{
	int answer=var1;

	if(answer>var2)
	{
		answer=var2;
	}

	return answer;
}



long int pow_int(int base, unsigned int exponent)
{
	long int answer=1;

	answer=(long int)pow((double)base, (double)exponent);

	return answer;
}



long int pow_int_mod(int base, unsigned int exponent, int mod)
{
	long int answer=1;
	while (exponent>0)
	{
		if (exponent&1)
		{
			answer=(answer*(base%mod))%mod;
			if(answer==0)
			{
				break;
			}
		}
		base=(base*base)%mod;
		exponent=exponent>>1;
	}
	return answer;
}



unsigned long int factorial(unsigned int num)
{
	unsigned int i=0;
	unsigned long int answer=1;
	rep(i, 2, num)
	{
		answer=answer*i;
	}
	return answer;
}



unsigned long int factorial_mod(unsigned int num, int mod)
{
	unsigned int i=0;
	unsigned long int answer=1;
	rep(i, 2, num)
	{
		answer=(answer*(i%mod))%mod;
		if(answer==0)
		{
			break;
		}
	}
	return answer;
}



unsigned long int combination(unsigned int n, unsigned int k)
{
	unsigned int i=0;
	unsigned long int numerator=1;
	unsigned long int denominator=1;

	k=min_int(k, n-k);

	rep(i, 2, k)
	{
		numerator=numerator*(n+1-i);
		denominator=denominator*i;
	}
	return numerator/denominator;
}



double combination_general(double alpha, unsigned int k)
{
	unsigned int i;
	double numerator=1;
	unsigned long int denominator=1;

	rep(i, 1, k)
	{
		numerator=numerator*(alpha+1-i);
		denominator=denominator*i;
	}
	return numerator/denominator;
}



int gcd(int a, int b)
{
	if(a<b)
	{
		swap_int(&a, &b);
	}

	int r=a%b;

	while(r!=0)
	{
		a=b;
		b=r;
		r=a%b;
	}

	return b;
}



int lcm(int a, int b)
{
	return (a/gcd(a, b))*b;
}



int order(long int num)
{
	int answer=-1;
	while(num!=0)
	{
		num=num/10;
		answer=answer+1;
	}
	return answer;
}



long int convert_adic_int(char num[], unsigned int p_adic, unsigned int q_adic)
{
	int i=0;
	long int answer=0;
	bool sgn=0;
	long int tmp=strtol(num, NULL, p_adic);

	if(tmp<0)
	{
		sgn=1;
		tmp=-tmp;
	}

	while(tmp>0)
	{
		answer=answer+(tmp%q_adic)*pow_int(10, i);
		tmp=tmp/q_adic;
		i=i+1;
	}

	if(sgn==0)
	{
		return answer;
	}
	else
	{
		return -answer;
	}
}



void convert_adic_char(char num[], unsigned int p_adic, unsigned int q_adic)
{
	int i=0;
	int mod[32]={0};
	bool sgn=0;
	int digit=0;
	long int tmp=strtol(num, NULL, p_adic);

	if(tmp<0)
	{
		sgn=1;
		tmp=-tmp;
	}

	while(tmp>0)
	{
		mod[i]=tmp%q_adic;

		digit=i;
		tmp=tmp/q_adic;
		i=i+1;
	}

	fill_char(num, 0, strlen(num)-1, '\0');

	if(sgn==1)
	{
		num[0]='-';
	}

	rep(i, 0, digit)
	{
		if(mod[i]<10)
		{
			num[sgn+digit-i]='0'+mod[i];
		}
		else if(mod[i]<q_adic)
		{
			num[sgn+digit-i]='a'+(mod[i]-10);
		}
	}
}



int count_mod(int min, int max, int num)
{
	if(min>0)
	{
		return max/num-(min-1)/num;
	}
	else if(min==0)
	{
		return max/num+1;
	}
	else
	{
		return -1;
	}
}



void shift_char(char array[], int min, int max, int num)
{
	int i;

	for(i=max; i>=min; i=i-1)
	{
		array[i+num]=array[i];
	}

	rep(i, min, min+num-1)
	{
		array[i]='\0';
	}
}



void convert_char(char array[], int min, int max, char pre, char post)
{
	int i=min;

	rep(i, min, max)
	{
		if(array[i]==pre)
		{
			array[i]=post;
		}
	}
}



void sort_asc_int(int array[], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(array[i]>array[j])
			{
				swap_int(&array[i], &array[j]);
			}
		}
	}
}



void sort_des_int(int array[], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(array[i]<array[j])
			{
				swap_int(&array[i], &array[j]);
			}
		}
	}
}



void sort_asc_char_dic(char array[], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(strcmp(&array[i], &array[j])>0)
			{
				swap_char(&array[i], &array[j]);
			}
		}
	}
}



void sort_asc_char_array_dic(int size, char array[][size], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(strcmp(array[i], array[j])>0)
			{
				swap_char_array(array[i], array[j], 0, size-1);
			}
		}
	}
}



void sort_des_char_dic(char array[], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(strcmp(&array[i], &array[j])<0)
			{
				swap_char(&array[i], &array[j]);
			}
		}
	}
}



void sort_des_char_array_dic(int size, char array[][size], int min, int max)
{
	int i, j;

	rep(i, min, max)
	{
		rep(j, i+1, max)
		{
			if(strcmp(array[i], array[j])<0)
			{
				swap_char_array(array[i], array[j], 0, size-1);
			}
		}
	}
}



int max_int_array(int array[], int min, int max)
{
	int i;
	int answer;

	answer=array[min];

	for(i=min+1; i<=max; i=i+1)
	{
		answer=max_int(answer, array[i]);
	}

	return answer;
}



void max_int_array_num(int array[], int min, int max, int answer[])
{
	int i=min;
	int count=0;
	int max_value=max_int_array(array, min, max);

	fill_int(answer, 0, max-min, -1);

	rep(i, min, max)
	{
		if(array[i]==max_value)
		{
			answer[count]=i;
			count=count+1;
		}
	}
}



int min_int_array(int array[], int min, int max)
{
	int i;
	int answer;

	answer=array[min];

	for(i=min+1; i<=max; i=i+1)
	{
		answer=min_int(answer, array[i]);
	}

	return answer;
}



void min_int_array_num(int array[], int min, int max, int answer[])
{
	int i=min;
	int count=0;
	int min_value=min_int_array(array, min, max);

	fill_int(answer, 0, max-min, -1);

	rep(i, min, max)
	{
		if(array[i]==min_value)
		{
			answer[count]=i;
			count=count+1;
		}
	}
}



int max_char_array_dic(int size, char array[][size], int min, int max)
{
	int i;
	int answer=min;

	rep(i, min+1, max)
	{
		if(strcmp(array[answer], array[i])<0)
		{
			answer=i;
		}
	}

	return answer;
}



int min_char_array_dic(int size, char array[][size], int min, int max)
{
	int i;
	int answer=min;

	rep(i, min+1, max)
	{
		if(strcmp(array[answer], array[i])>0)
		{
			answer=i;
		}
	}

	return answer;
}



int sum_array(int array[], int min, int max)
{
	int i=min;
	int answer=0;

	rep(i, min, max)
	{
		answer=answer+array[i];
	}

	return answer;
}



bool detect_int(int array[], int min, int max, int element)
{
	int i;
	bool answer=0;

	if_exists(i, min, max, array[i]==element)
	{
		answer=1;
	}

	return answer;
}



bool detect_char(char array[], int min, int max, char element)
{
	int i;
	bool answer=0;

	if_exists(i, min, max, array[i]==element)
	{
		answer=1;
	}
	return answer;
}



bool detect_char_array(char array[], int min, int max, char element[], int start, int goal)
{
	int i, j;
	bool answer=0;

	if(goal-start<=max-min)
	{
		rep(i, min, max-(goal-start))
		{
			if_forall(j, start, goal, array[i+j-start]==element[j])
			{
				answer=1;
				break;
			}
		}
	}

	return answer;
}




int count_differentelement_int(int array[], int min, int max)
{
	int i=min;
	int pickup[max];
	fill_int(pickup, 0, max, 0);
	int answer=0;

	rep(i, min, max)
	{
		if(detect_int(pickup, 0, answer-1, array[i])==0)
		{
			pickup[answer]=array[i];
			answer=answer+1;
		}
	}

	return answer;
}



int count_differentelement_char(char array[], int min, int max)
{
	int i=min;
	char pickup[max];
	fill_char(pickup, 0, max, '\0');
	int answer=0;

	rep(i, min, max)
	{
		if(detect_char_array(pickup, 0, answer-1, array, i, i)==0)
		{
			pickup[answer]=array[i];
			answer=answer+1;
		}
	}

	return answer;
}



int count_equalelement_int(int array[], int min, int max, int element)
{
	int i;
	int answer=0;

	rep(i, min, max)
	{
		if(array[i]==element)
		{
			answer=answer+1;
		}
	}

	return answer;
}



int count_equalelement_char(char array[], int min, int max, char element)
{
	int i;
	int answer=0;

	rep(i, min, max)
	{
		if(array[i]==element)
		{
			answer=answer+1;
		}
	}

	return answer;
}



int count_equalelement_char_array(int size, char array[][size], int min, int max, char element[])
{
	int i;
	int answer=0;

	rep(i, min, max)
	{
		if(strcmp(array[i], element)==0)
		{
			answer=answer+1;
		}
	}
	return answer;
}



void scanf_int_array(int array[], int min, int max)
{
	int i;
	rep(i, min, max)
	{
		scanf("%d", &array[i]);
	}
}



void scanf_int_array_2(int array1[], int array2[], int min, int max)
{
	int i;
	rep(i, min, max)
	{
		scanf("%d %d", &array1[i], &array2[i]);
	}
}



void scanf_int_array_3(int array1[], int array2[], int array3[], int min, int max)
{
	int i;
	rep(i, min, max)
	{
		scanf("%d %d %d", &array1[i], &array2[i], &array3[i]);
	}
}



void scanf_char_array(char array[], int num)
{
	scanf("%s", array);
	shift_char(array, 0, strlen(array)-1, num);
}



void printf_int_array(int array[], int min, int max)
{
	int i;
	rep(i, min, max)
	{
		printf("%d\n", array[i]);
	}
}



void printf_char_array(char array[], int min, int max)
{
	int i;
	for(i=min; i<=max; i=i+1)
	{
		printf("%c", array[i]);
	}
}












int main()
{
	int i;
	int N, M, X;
	scanf("%d %d %d", &N, &M, &X);

	int A[M+1];
	scanf_int_array(A, 1, M);

	int count=0;
	rep(i, 1, M)
	{
		if(A[i]<X)
		{
			count=count+1;
		}
		else
		{
			break;
		}
	}
	printf("%d\n", min_int(count, M-count));
	return 0;
}

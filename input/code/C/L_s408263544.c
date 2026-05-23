
char pStr[DEF_ELM_MAX];

typedef struct _PAIR {
	int indx;
	int diff;
} PAIR;

int S1[DEF_ELM_MAX];
PAIR S2[DEF_ELM_MAX];

int s1_top = 0;

int s1_pop()
{
	return S1[--s1_top];
}

void s1_push(int x)
{
	S1[s1_top++] = x;
}

int s2_top = 0;

PAIR s2_Top()
{
	return S2[s2_top-1];
}

PAIR s2_pop()
{
	return S2[--s2_top];
}

void s2_push(int indx, int diff)
{
	PAIR x;
	x.indx = indx;
	x.diff = diff;

	S2[s2_top++] = x;
}

/////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////
int main(void)
{
	int j;
	int i;
	int sum;
	int a;
	char c;

//
	scanf("%s", pStr);

	sum = 0;
	i = 0;
	while ( (c = pStr[i]) != 0 ) {
		if ( c == '\\' ) {
			s1_push(i);
		} else if ( s1_top > 0 && c == '/') {
			j = s1_pop();
			a = i - j;
			sum += a;
			while ( s2_top > 0 && s2_Top().indx > j) {
				a += s2_pop().diff;
			}
			s2_push(j, a);
		}
		i++;
	}

	printf("%d\n", sum);
	printf("%d", s2_top);
	for ( i = 0 ; i < s2_top; i++ ) {
		printf(" ");
		printf("%d", S2[i].diff);	
	}
	printf("\n");	

	return 0;
}

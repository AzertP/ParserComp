#include <stdio.h> 

int main(void)
{
	int x,i;

	for(i=0; i<10000; i++)
	{
		
		scanf("%d",&x);

		if(x==0)
		{
			break;
		}

		printf("Case %d: %d\n",i+1, x);

	}


	return 0;
}
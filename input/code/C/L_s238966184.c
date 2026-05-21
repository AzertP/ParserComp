#include <stdio.h>
#include <stdlib.h>

typedef struct Bomb
{
	int map[8][8];
	int x, y;
} BOMB;

int x_cnt, y_cnt;

BOMB* explode(BOMB*);	/*Recursion*/

int main()
{
    int input_times, input_cnt;
    char dump;
    BOMB input[50];
    
    /*Read input times*/
    scanf("%d", &input_times);
    input_cnt = 0;
    while(input_cnt < input_times)
    {
		/*Read input map*/
		dump = getchar();
		for(y_cnt = 0; y_cnt < 8; y_cnt++)
		{
			dump = getchar();
			for(x_cnt = 0; x_cnt < 8; x_cnt++)
			{
				input[input_cnt].map[x_cnt][y_cnt] = getchar() - 48;
			}
		}
		/*Read first explosion*/
		scanf("%d", &(input[input_cnt].x));
		scanf("%d", &(input[input_cnt].y));
		
		/*Process*/
		input[input_cnt].map[(input[input_cnt].x) - 1][(input[input_cnt].y) - 1] = 2;
		explode(&input[input_cnt]);
		
		input_cnt++;
	}
    
    input_cnt = 0;
    while(input_cnt < input_times)
    {
    	printf("Data %d:\n", input_cnt + 1);
    	for(y_cnt = 0; y_cnt < 8; y_cnt++)
		{
			for(x_cnt = 0; x_cnt < 8; x_cnt++)
			{
				printf("%d", input[input_cnt].map[x_cnt][y_cnt]);
			}
			printf("\n");
		}
    	input_cnt++;
	}

    
    return 0;
} 

BOMB* explode(BOMB* bombmap)
{
	int explode_cnt;
	for(y_cnt = 0; y_cnt < 8; y_cnt++)
	{
		for(x_cnt = 0; x_cnt < 8; x_cnt++)
		{
			if(bombmap->map[x_cnt][y_cnt] == 2)
			{
				bombmap->map[x_cnt][y_cnt] = 0;
				
				explode_cnt = 1;
				while((x_cnt - explode_cnt) >= 0 && explode_cnt <= 3)
				{
					if((bombmap->map[x_cnt - explode_cnt][y_cnt]) == 1)
					{
						(bombmap->map[x_cnt - explode_cnt][y_cnt]) = 2;
					}
					explode_cnt++;
				}
				
				explode_cnt = 1;
				while((x_cnt + explode_cnt) <= 7 && explode_cnt <= 3)
				{
					if((bombmap->map[x_cnt + explode_cnt][y_cnt]) == 1)
					{
						(bombmap->map[x_cnt + explode_cnt][y_cnt]) = 2;
					}
					explode_cnt++;
				}
				
				explode_cnt = 1;
				while((y_cnt - explode_cnt) >= 0 && explode_cnt <= 3)
				{
					/*printf("3: x:%d y:%d e:%d\n", x_cnt, y_cnt, explode_cnt);*/
					if((bombmap->map[x_cnt][y_cnt - explode_cnt]) == 1)
					{
						(bombmap->map[x_cnt][y_cnt - explode_cnt]) = 2;
					}
					explode_cnt++;
				}
				
				explode_cnt = 1;
				while((y_cnt + explode_cnt) <= 7 && explode_cnt <= 3)
				{
					/*printf("4: x:%d y:%d e:%d\n", x_cnt, y_cnt, explode_cnt);*/
					if((bombmap->map[x_cnt][y_cnt + explode_cnt]) == 1)
					{
						(bombmap->map[x_cnt][y_cnt + explode_cnt]) = 2;
					}
					explode_cnt++;
				}
				/*printf("LOOP\n");*/
				return explode(bombmap);
			}
		}
	}
	
	return bombmap;
}
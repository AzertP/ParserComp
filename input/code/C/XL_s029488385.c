#include<stdio.h>

#define SIZE 10

#define compile 0
/*
デバッグ用
0:実装 1:main関数 2:initialize関数 3:check_bomb関数 4:tnt関数 5:little_boy関数 6:fat_man関数 7:check_result関数
*/

void initialize(int array[][SIZE]);
void check_bomb(int map[][SIZE], int input[]);
void tnt(int map[][SIZE], int input[]);
void little_boy(int map[][SIZE], int input[]);
void fat_man(int map[][SIZE], int input[]);
void check_result(int map[][SIZE], int *white,int *max);

main()
{
	int map[SIZE][SIZE];
	int input[3] = {0, 0, 0};
	int white = 0, max = 0;
	initialize(map);
	while(scanf("%d,%d,%d", &input[0], &input[1], &input[2]) != EOF)
	{
		check_bomb(map, input);
#if compile == 1
		int i = 0, j = 0;
		for(i = 0; i < SIZE; i++)
		{
			for(j = 0; j < SIZE; j++)
			{
				printf("%d ", map[i][j]);
			}
			printf("\n");
		}
#endif
	}
	check_result(map, &white, &max);
	printf("%d\n%d\n", white, max);
	return 0;
}
void initialize(int array[][SIZE])
{
	short i = 0, j = 0;
	for(i = 0; i < SIZE; i++)
	{
		for(j = 0; j < SIZE; j++)
		{
			array[i][j] = 0;
		}
	}
}
void check_bomb(int map[][SIZE], int input[])
{
	if((input[0] < 0) || (input[0] >= 10) || (input[1] < 0) || (input[1] >= 10))
	{
		return;
	}
	else
	{
		if(input[2] == 1)
		{
#if compile == 3
			printf("call tnt\n");
#endif
			tnt(map, input);
		}
		else if(input[2] == 2)
		{
#if compile == 3
			printf("call little_boy\n");
#endif
			little_boy(map, input);
		}
		else if(input[2] == 3)
		{
#if compile == 3
			printf("call fat_man\n");
#endif
			fat_man(map, input);
		}
	}
}
void tnt(int map[][SIZE], int input[])
{
	short i = 0, j = 0;
	for(i = input[0] - 1; i <= input[0] + 1; i++) //x軸
	{
		if((i < 0) || (i > 10))
		{
			continue;
		}
		else
		{
			map[input[1]][i]++;
		}
#if compile == 4
		printf("map[%d][%d] = %d\n", input[1], i, map[input[1]][i]);
#endif
	}
	for(i = input[1] - 1; i <= input[1] + 1; i++) //y軸
	{
		if((i < 0) || (i > 10))
		{
			continue;
		}
		else
		{
			map[i][input[0]]++;
		}
#if compile == 4
		printf("map[%d][%d] = %d\n", i, input[0], map[i][input[0]]);
#endif
	}
	map[input[1]][input[0]]--;
}
void little_boy(int map[][SIZE], int input[])
{
	short i = 0, j = 0;
	for(i = input[1] - 1; i <= input[1] + 1; i++)
	{
		for(j = input[0] - 1; j <= input[0] + 1; j++)
		{
			if((i < 0) || (j < 0) || (i >= 10) || (j >= 10))
			{
				continue;
			}
			else
			{
				map[i][j]++;
#if compile == 5
				printf("map[%d][%d] = %d\n", i, j, map[i][j]);
#endif
			}
		}
	}
}
void fat_man(int map[][SIZE], int input[])
{
	short i = 0, j = 0;
	for(i = input[1] - 1; i <= input[1] + 1; i++)
	{
		for(j = input[0] - 1; j <= input[0] + 1; j++)
		{
			if((i < 0) || (j < 0) || (i >= 10) || (j >= 10))
			{
				continue;
			}
			else
			{
				map[i][j]++;
#if compile == 6
				printf("map[%d][%d] = %d\n",i, j, map[i][j]);
#endif
			}
		}
	}
	if((input[0] - 2) >= 0)
	{
		map[input[1]][input[0] - 2]++;
#if compile == 6
		printf("map[%d][%d] = %d\n",input[1], input[0] - 2, map[input[1]][input[0] - 2]);
#endif
	}
	if((input[0] + 2) <= 9)
	{
		map[input[1]][input[0] + 2]++;
#if compile == 6
		printf("map[%d][%d] = %d\n",input[1], input[0] + 2, map[input[1]][input[0] + 2]);
#endif
	}
	if((input[1] - 2) >= 0)
	{
		map[input[1] - 2][input[0]]++;
#if compile == 6
		printf("map[%d][%d] = %d\n", input[1] - 2, input[0], map[input[1] - 2][input[0]]);
#endif
	}
	if((input[1] + 2) <= 9)
	{
		map[input[1] + 2][input[0]]++;
#if compile == 6
		printf("map[%d][%d] = %d\n", input[1] + 2, input[0], map[input[1] + 2][input[0]]);
#endif
	}
}
void check_result(int map[][SIZE], int *white,int *max)
{
	short i = 0, j = 0;
	*max = map[0][0];
	for(i = 0; i < SIZE; i++)
	{
		for(j = 0; j < SIZE; j++)
		{
			if(map[i][j] > *max)
			{
				*max = map[i][j];
			}
			if(map[i][j] == 0)
			{
				*white += 1;
			}
		}
	}
}
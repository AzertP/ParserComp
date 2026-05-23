

/*

0: 1:main 2:initialize 3:check_bomb 4:tnt 5:little_boy 6:fat_man 7:check_result
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
		int i = 0, j = 0;
		for(i = 0; i < SIZE; i++)
		{
			for(j = 0; j < SIZE; j++)
			{
				printf("%d ", map[i][j]);
			}
			printf("\n");
		}
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
			printf("call tnt\n");
			tnt(map, input);
		}
		else if(input[2] == 2)
		{
			printf("call little_boy\n");
			little_boy(map, input);
		}
		else if(input[2] == 3)
		{
			printf("call fat_man\n");
			fat_man(map, input);
		}
	}
}
void tnt(int map[][SIZE], int input[])
{
	short i = 0, j = 0;
	for(i = input[0] - 1; i <= input[0] + 1; i++) //x
	{
		if((i < 0) || (i > 10))
		{
			continue;
		}
		else
		{
			map[input[1]][i]++;
		}
		printf("map[%d][%d] = %d\n", input[1], i, map[input[1]][i]);
	}
	for(i = input[1] - 1; i <= input[1] + 1; i++) //y
	{
		if((i < 0) || (i > 10))
		{
			continue;
		}
		else
		{
			map[i][input[0]]++;
		}
		printf("map[%d][%d] = %d\n", i, input[0], map[i][input[0]]);
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
				printf("map[%d][%d] = %d\n", i, j, map[i][j]);
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
				printf("map[%d][%d] = %d\n",i, j, map[i][j]);
			}
		}
	}
	if((input[0] - 2) >= 0)
	{
		map[input[1]][input[0] - 2]++;
		printf("map[%d][%d] = %d\n",input[1], input[0] - 2, map[input[1]][input[0] - 2]);
	}
	if((input[0] + 2) <= 9)
	{
		map[input[1]][input[0] + 2]++;
		printf("map[%d][%d] = %d\n",input[1], input[0] + 2, map[input[1]][input[0] + 2]);
	}
	if((input[1] - 2) >= 0)
	{
		map[input[1] - 2][input[0]]++;
		printf("map[%d][%d] = %d\n", input[1] - 2, input[0], map[input[1] - 2][input[0]]);
	}
	if((input[1] + 2) <= 9)
	{
		map[input[1] + 2][input[0]]++;
		printf("map[%d][%d] = %d\n", input[1] + 2, input[0], map[input[1] + 2][input[0]]);
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

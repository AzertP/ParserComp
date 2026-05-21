#include<stdio.h>

int shori(int,int);
int data[101][101], suuji[101];

int main(){

  int i, j, n;

  scanf("%d",&n);
  
  for(i = 0; i < n; i++)
    {
      if(i != 0)data[i][i] = 0;
      scanf("%d%d",&suuji[i],&suuji[i+1]);
    }

  for(j = 1; j <= n; j++)
    {
      for(i = 1; i <= n - j; i++)
	{
	  data[i][i+j] = shori(i,i+j);
	}
    }
  printf("%d\n",data[1][n]);

  return 0;
}

int shori(int x, int y)
{
  int i, min = 0, temp;
  for(i = x; i < y; i++)
    {
      temp = data[x][i] + data[i + 1][y] + (suuji[x - 1] * suuji[i] * suuji[y]);
      if(i == x || temp < min)min = temp;
    }
  return min;
}


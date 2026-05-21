#include <stdio.h>

void change(int,int);
int muki = 0,flag = 0;

int main(){
	int i,j,x = 0,y = 0,wall_x[4][5],wall_y[5][4];
	for(i = 0;i < 9;i++){
		if(i % 2 == 0) for(j = 0;j < 4;j++) scanf("%1d",&wall_x[j][i / 2]);
		else for(j = 0;j < 5;j++) scanf("%1d",&wall_y[j][(i - 1)/ 2]);
	}
	do{
		if(muki == 0){
			if(x != 4 && wall_x[x][y] == 1){
				flag = 0;
				change(x,y);
				putchar('R');
				x++;
			}else change(x,y);
		}else if(muki == 1){
			if(y != 4 && wall_y[x][y] == 1){
				flag = 0;
				change(x,y);
				putchar('D');
				y++;
			}else change(x,y);
		}else if(muki == 2){
			if(x != 0 && wall_x[x - 1][y] == 1){
				flag = 0;
				change(x,y);
				putchar('L');
				x--;
			}else change(x,y);
		}else if(muki == 3){
			if(y != 0 && wall_y[x][y - 1] == 1){
				flag = 0;
				change(x,y);
				putchar('U');
				y--;
			}else change(x,y);
		}
	}while(x != 0 || y != 0);
	puts("");
	return 0;
}

void change(int x,int y){
	int i;
	if(!flag){
		flag = 1;
		muki = (muki + 3) % 4;
	}
	else muki = (muki + 1) % 4;
}
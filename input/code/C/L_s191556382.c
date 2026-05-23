void N(int dice[]);
void E(int dice[]);
void S(int dice[]);
void W(int dice[]);
void turn(int dice[]);
int main(void){
	int dice1[6],dice2[6],dice3[6];
	int i,j,flag=0;
	scanf("%d %d %d %d %d %d",&dice1[0],&dice1[1],&dice1[2],&dice1[3],&dice1[4],&dice1[5]);
	scanf("%d %d %d %d %d %d",&dice2[0],&dice2[1],&dice2[2],&dice2[3],&dice2[4],&dice2[5]);
	for(i=0;i<6;i++){
		if(dice1[0]==dice2[i]){
			memcpy(dice3,dice2,sizeof(int)*6);
			switch(i){
			case 0:
				break;
			case 1:
				N(dice3);
				break;
			case 2:
				W(dice3);
				break;
			case 3:
				E(dice3);
				break;
			case 4:
				S(dice3);
				break;
			case 5:
				N(dice3);
				N(dice3);
				break;
			}
			for(j=0;j<4;j++){
				if(dice1[0]==dice3[0] && dice1[1]==dice3[1] && dice1[2]==dice3[2] && dice1[3]==dice3[3] && dice1[4]==dice3[4] && dice1[5]==dice3[5]){
					flag=1;
					break;
				}
				else turn(dice3);
			}
		}
		if(flag==1) break;
	}
	if(flag==1) printf("Yes\n");
	else printf("No\n");

	return 0;
}
void N(int dice[]){
	int tmp;
	tmp=dice[0];
	dice[0]=dice[1];
	dice[1]=dice[5];
	dice[5]=dice[4];
	dice[4]=tmp; 
}
void E(int dice[]){
	int tmp;
	tmp=dice[0];
	dice[0]=dice[3];
	dice[3]=dice[5];
	dice[5]=dice[2];
	dice[2]=tmp;
}
void S(int dice[]){
	int tmp;
	tmp=dice[0];
	dice[0]=dice[4];
	dice[4]=dice[5];
	dice[5]=dice[1];
	dice[1]=tmp;
}
void W(int dice[]){
	int tmp;
	tmp=dice[0];
	dice[0]=dice[2];
	dice[2]=dice[5];
	dice[5]=dice[3];
	dice[3]=tmp;
}
void turn(int dice[]){
	int tmp;
	tmp=dice[1];
	dice[1]=dice[2];
	dice[2]=dice[4];
	dice[4]=dice[3];
	dice[3]=tmp;
}


#include <stdio.h>
#include <math.h>
int main()
{
	int tanka[100];
	int kosuu[100];
	int i=0;
	int sum=0;
	int kazu=0;
	int ret;
	int count;
	double heikin=0;
	while(1){
		ret=scanf("%d,%d",&tanka[i],&kosuu[i]);    //tanka[i]とkosuu[i]に数字を入れる；
		if(ret==EOF){
			break;
		}
		i++;
	}
	count=i;
	for(i=0; i<count; i++){
		sum+=tanka[i]*kosuu[i];       //合計金額の計算
		kazu+=kosuu[i];               //数量合計の計算（平均計算のため）
	}
	heikin=(double)kazu/(double)count;    //数量平均の計算
	heikin=floor(heikin+0.5);         //平均値を四捨五入する（0.5を加えて、小数点以下を切り捨てる）
	kazu=(int)heikin;                 //heikin（実数）をkazu（整数）に入れる。小数点以下は既に四捨五入してある。
	printf("%d\n",sum);     //合計金額の表示
	printf("%d\n",kazu);    //数量平均の表示
	return 0;
}
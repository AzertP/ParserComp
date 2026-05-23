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
		ret=scanf("%d,%d",&tanka[i],&kosuu[i]);    //tanka[i]kosuu[i]
		if(ret==EOF){
			break;
		}
		i++;
	}
	count=i;
	for(i=0; i<count; i++){
		sum+=tanka[i]*kosuu[i];       //
		kazu+=kosuu[i];               //
	}
	heikin=(double)kazu/(double)count;    //
	heikin=floor(heikin+0.5);         //0.5
	kazu=(int)heikin;                 //heikinkazu
	printf("%d\n",sum);     //
	printf("%d\n",kazu);    //
	return 0;
}

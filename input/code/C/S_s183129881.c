int main()
{
  int i,n,k,atari,total,kai;

  for(i=0; i<10000; i++){
    scanf("%d",&n);

    if(n==0) break;

      kai=n/4;
      total=0;
      
      for(k=0; k<kai; k++){
	scanf("%d",&atari);

	total+=atari;
    }
    printf("%d\n",total);
  }
  return 0;
}

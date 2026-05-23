int main(void){
   int nowTime;
   int later;
   int startTime;
   scanf("%d", &nowTime);
   scanf("%d", &later);
   if( nowTime >= 24)
      nowTime = nowTime - 24;
   if( later >= 24 )
      later = later - 24;
   startTime = nowTime + later;
   if (startTime >= 24)
      startTime = startTime - 24;
   printf("%d" , startTime);
   return 0;
}

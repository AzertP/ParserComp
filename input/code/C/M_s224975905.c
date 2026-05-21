#include<stdio.h>
 int main(){
   int num,i,j,z;
   int key;
   scanf("%d",&num);
   int num2[num];
   
   for(i=0;i<num;i++){
     scanf("%d",&num2[i]);
   }
   
   for(z=0;z<num;z++){
     if(z == num-1){
       printf("%d\n",num2[z]);
     }
     else  {printf("%d ",num2[z]);}
   }
   for(i=1;i<num;i++){
     key = num2[i];
     j = i-1;
     while(j>=0 && num2[j] > key){
       num2[j+1] = num2[j];
       j--;
       num2[j+1] = key;
     }
     for(z=0;z<num;z++){
       if(z == num-1){
	 printf("%d\n",num2[z]);
       }
	 else {printf("%d ",num2[z]);}
       }
     }
   return 0;
 }


   
     


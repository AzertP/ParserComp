import java.util.Scanner;
public class Main {
 private static Scanner sc;
 public static void main(String[] args){
  int i,x;
  sc = new Scanner(System.in);
  int n = sc.nextInt();
  for(i=3;i<=n;i++){
	  x=i;
	   if(n<3||n>10000)break;
 
	   if(x%3==0){
  	     System.out.print(" "+i);continue;
     }else if(x%10==3){
  	     System.out.print(" "+i);continue;
     }
      
      if(x>10 && x<=100){
    	  x=i;
          x/=10;
          if(x==3 && x%10!=0){                
       	   System.out.print(" "+i);continue;  
          }
      }
      
      else if(x>100 && x<=1000){
   	      x=i;
          x=x-(x/100)*100;                    
          x/=10;
          if(x==3 && x%10!=0){
       	   System.out.print(" "+i);continue;
          }
          x=i;
          x/=100;
          if(x==3 && x%100!=0){
       	   System.out.print(" "+i);continue;
          }
      }
      
       
      else if(x>1000 && x<=10000){
   	      x=i;
          x=x-(x/1000)*1000;
          x=x-(x/100)*100;
          x/=10;
          if(x==3 && x%10!=0){
       	   System.out.print(" "+i);continue;
          }
          x=i;
          x=x-(x/1000)*1000;
          x/=100;
          if(x==3 && x%100!=0){
       	   System.out.print(" "+i);continue;
          }
          x=i;
          x/=1000;
          if(x==3 && x%1000!=0){
       	   System.out.print(" "+i);continue;
          }
      }
  }
  
  System.out.println();
 }
}
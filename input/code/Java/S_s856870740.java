import java.util.*;
public class Main{
	public static void main(String[] args){
    	Scanner sc = new Scanner(System.in);
      int a = sc.nextInt();
      int b = sc.nextInt();
      int x1 = 0;
      int x2 = 0;
      int x3 = 0;
      int x4 = 0;
      int x5 = 0;
      int count = 0;
      for(int i = a;i <= b;i++){
      	x5 = i%10;
        x1 = i/10000;
        x2 = (i-x1*10000)/1000;
        x3 = (i-x1*10000-x2*1000)/100;
        x4 = (i-x1*10000-x2*1000-x3*100)/10;
        if(x1==x5&&x2==x4)count++;
      }
      System.out.println(count);
    }
}

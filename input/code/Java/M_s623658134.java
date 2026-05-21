import java.util.Scanner;
import java.util.ArrayList;
 
public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
      ArrayList<String> list = new ArrayList<String>();
      int x_1,x_2,x_3,x_4,x_5;
      int i = 0;
      String an = scan.nextLine();
      String atai[] = an.split(" ",0);
      int num[] = new int[2];
      while(i < 2){
        num[i] = Integer.parseInt(atai[i]);
        i++;
      }
      x_1 = num[0];
      x_2 = num[1];
//      System.out.println(A+","+B+","+H+","+M);
      check(num);
        scan.close();
    }
  
  public static void check(int[] num){
   //System.out.println(num[0]);
    int X, Y;
    int s;
    X = num[0];
    Y = num[1];
     s= Y%2;
    if(X*4 < Y){
      System.out.println("No");
    }else if( s != 0){
      System.out.println("No");
    }else if(X*2 > Y){
      System.out.println("No");
    }else{
      System.out.println("Yes");
    }
   return;  
  }
}

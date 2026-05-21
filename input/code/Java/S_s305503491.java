
import java.util.Scanner;

public class Main {

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    String S = sc.next();

    for (int i = 0; i < S.length(); i++) {
      char c= S.charAt(i);
      if( (i+1)%2 ==0){
        if(c != 'L' && c!='U' && c!='D'){
          System.out.println("No");
          return;
        }
      }else{
        if(c != 'R' && c!='U' && c!='D'){
          System.out.println("No");
          return;
        }
      }
    }
    System.out.println("Yes");
    }

}

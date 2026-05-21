import java.util.*;

public class Main{
  public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    String d = sc.next();
    String s[] = d.split("/");
    System.out.println(Integer.parseInt(s[1])<5 ? "Heisei" : "TBD");
  }
}
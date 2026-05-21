import java.util.*;
public class Main{
   public static void main(String[] args) {
     Scanner sc = new Scanner(System.in);
     long n = sc.nextLong();
     long k = sc.nextLong();
     long i = n%k;
     long ans = Math.min(i,k-i);
     System.out.println(ans);
   }
}

import java.util.*;

public class Main {
    public static void main(String args[]) {
        
        Scanner sc = new Scanner(System.in);
        
        int n = Integer.parseInt(sc.next());
        int k = Integer.parseInt(sc.next());
        
        int counter = 0;
        int tmp = n;
        while(tmp > 0) {
            tmp = tmp / k;
            ++counter;
        }
        
        System.out.println(counter);
        
    }
}
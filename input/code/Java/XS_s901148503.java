import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int x = sc.nextInt();
        int t = sc.nextInt();
        int sand = x - t;
        if (sand < 0 ) {
            System.out.println(0);
        } else {
            System.out.println(sand);
        }
    }
}
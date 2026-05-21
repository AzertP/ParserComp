import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        int a = Integer.parseInt(sc.next());
        int b = Integer.parseInt(sc.next());
        if (b >= a) {
            System.out.println(a);
        } else {
            System.out.println(a - 1);
        }
    }
}

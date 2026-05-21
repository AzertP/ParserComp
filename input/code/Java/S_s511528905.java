import java.util.*;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        int c = sc.nextInt();

        int ab = a + b;
        int bc = b + c;
        int ca = c + a;

        System.out.println(Math.min(ca, Math.min(ab, bc)));



    }


}
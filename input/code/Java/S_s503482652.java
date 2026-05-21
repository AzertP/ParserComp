import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        new Main();
    }

    public Main() {
        Scanner sc = new Scanner(System.in);
        String S = sc.next();
        String T = sc.next();
        int min = Integer.MAX_VALUE;
        for (int i = 0; i <= S.length() - T.length(); i++) {
            int rewrite = 0;
            for (int j = 0; j < T.length(); j++) {
                if (T.charAt(j) != S.charAt(i + j)) rewrite++;
            }
            min = Math.min(min, rewrite);
        }
        System.out.println(min);
    }
}

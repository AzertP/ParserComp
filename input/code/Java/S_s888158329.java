import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner cin = new Scanner(System.in);
        int N = cin.nextInt();
        int T = cin.nextInt();
        int A = cin.nextInt();
        int[] H = new int[N];

        for (int i = 0; i < N; i++) {
            H[i] = cin.nextInt();
        }

        double min = Math.abs(A - (T - H[0] * 0.006));
        int idx = 0;
        for (int i = 1; i < N; i++) {
            double d = Math.abs(A - (T - H[i] * 0.006));
            if (min > d) {
                min = d;
                idx = i;
            }
        }
        System.out.println(idx + 1);
    }
}
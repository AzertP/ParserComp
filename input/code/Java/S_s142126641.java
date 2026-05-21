import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()) {
            int[] l = new int[8];
            for (int i = 0; i < 8; i++) {
                l[i] = sc.nextInt();
            }
            int v = 0;
            int min = Integer.MAX_VALUE;
            int[] c = { 1, 2, 1, 2, 1, 4, 1, 4 };
            int[] o = { 0, 2, 6, 4, 1, 3, 7, 5 };
            for (int i = 0; i < 8; i++) {
                int sum = 0;
                for (int j = 0; j < 8; j++) {
                    sum += Math.max(0, c[(o[i] + j) % 8] - l[j]);
                }
                if (min > sum) {
                    min = sum;
                    v = o[i];
                }
            }
            for (int i = 0; i < 8; i++) {
                System.out.print(c[(i + v) % 8] + (i != 7 ? " " : "\n"));
            }
        }
    }
}

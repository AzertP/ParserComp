import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner stdIn = new Scanner(System.in);
         
        int r = stdIn.nextInt();
        int c = stdIn.nextInt();
         
        int[][] h = new int[r+1][c+1];
         
        for(int i = 0; i < r; i++) {
            int rsum = 0;
            for(int j = 0; j < c; j++) {
                int tmp = stdIn.nextInt();
                rsum += tmp;
                h[i][j] = tmp;
            }
            h[i][c] = rsum;
        }
         
        for(int i = 0; i < c+1; i++) {
            int rsum = 0;
            for(int j = 0; j < r; j++) {
                rsum += h[j][i];
            }
            h[r][i] = rsum;
        }
         
        for(int i = 0; i < r+1; i++) {
            for(int j = 0; j < c; j++) {
                System.out.print(h[i][j] + " ");
            }
            System.out.println(h[i][c]);
        }
    }
}

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n  = sc.nextInt();
        int[] a = new int[n+1];
        long ans = 0;
        for (int i = 0; i < n; i++) {
            int x = sc.nextInt();
            if(x > n) ans++;
            else a[x]++;
        }
        for (int i = 1; i <= n; i++) {
            if(a[i] < i) ans += a[i];
            else ans += (long)a[i] - i;
        }

        System.out.println(ans);
        sc.close();

    }

}

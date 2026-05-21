import java.util.*;

public class Main {

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    int t = sc.nextInt();
    int a = sc.nextInt();
    double[] data = new double[n];

    for (int i = 0; i < n; i++) {
      data[i] = sc.nextInt();
      data[i] = Math.abs(a-(t-data[i]*0.006));
    }

    double min = data[0];
    int m = 0;

    for (int i = 1; i < n; i++) {
      if (min > data[i]) {
        min = data[i];
        m = i;
      }
    }

    System.out.println(m+1);

  }

}

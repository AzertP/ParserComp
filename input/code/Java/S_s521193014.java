import java.util.Arrays;
import java.util.Scanner;
import java.util.HashMap;

public class Main {

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    HashMap<String, Long> s = new HashMap<>();
    long count = 0;

    for (int i = 0; i < n; ++i) {
      char[] ci = sc.next().toCharArray();
      Arrays.sort(ci);
      String si = new String(ci);
      s.put(si, s.getOrDefault(si, 0L) + 1);
    }

    for (long v : s.values()) {
      count += v * (v - 1) / 2;
    }

    System.out.println(count);

  }
}
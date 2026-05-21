import java.util.*;
public class Main {
  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    int m = sc.nextInt();
    Integer[] nums = new Integer[n + 1];
    for (int i = 0; i < m; i++) {
      int s = sc.nextInt();
      int c = sc.nextInt();
      if (nums[s] != null && nums[s] != c) {
        System.out.println(-1);
        return;
      }
      nums[s] = c;
    }
    if (n == 1) {
      System.out.println(nums[1] == null ? 0 : nums[1]);
      return;
    }
    if (nums[1] != null && nums[1] == 0) {
      System.out.println(-1);
      return;
    }
    StringBuilder sb = new StringBuilder();
    if (nums[1] == null) sb.append(1);
    else sb.append(nums[1]);
    for (int i = 2; i <= n; i++) {
      if (nums[i] == null) sb.append(0);
      else sb.append(nums[i]);
    }
    System.out.println(sb.toString());
  }
}

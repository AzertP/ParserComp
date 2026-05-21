import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        final Scanner sc = new Scanner(System.in);

        final int n = sc.nextInt();
        final int m = sc.nextInt();

        final int[] p = new int[m];
        final String[] s = new String[m];
        Map<Integer, int[]> map = new HashMap<>();
        for (int i = 0; i < m; i++) {
            p[i] = sc.nextInt();
            s[i] = sc.next();
            if (!map.containsKey(p[i])) {
                map.put(p[i], new int[2]);
            }
            if ("AC".equals(s[i])) {
                map.put(p[i], new int[]{ ++map.get(p[i])[0], map.get(p[i])[1] });
            }
            if ("WA".equals(s[i]) && map.get(p[i])[0] == 0) {
                map.put(p[i], new int[]{map.get(p[i])[0], ++map.get(p[i])[1]});
            }
        }

        int ok = 0;
        int ng = 0;
        for (Map.Entry<Integer, int[]> entry: map.entrySet()) {
            if (entry.getValue()[0] > 0) {
                ok++;
                ng += entry.getValue()[1];
            }
        }

        System.out.println(ok + " " + ng);
    }
}

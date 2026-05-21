import java.util.*;

public class Main {

    public static void main(String args[]) {

        // 入力
        Scanner sc = new Scanner(System.in);
        int n = Integer.parseInt(sc.next());
        int k = Integer.parseInt(sc.next());
        Long[] a = new Long[n];
        for (int i = 0; i < n; i++) {
            a[i] = Long.parseLong(sc.next());
        }
        sc.close();

        // 主処理
        List<String> list = new ArrayList<>();
        for (int i = k; i < n; i++) {
            long first = a[i - k];
            long last = a[i];
            String s = first < last ? "Yes" : "No";
            list.add(s);
        }
        String result = String.join("\r\n", list);

        // 出力
        System.out.println(result);
    }
}


import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        Scanner stdin = new Scanner(System.in);
        int n = Integer.parseInt(stdin.nextLine());
        Map<Integer, Long> d = Pattern.compile(" ")
                                      .splitAsStream(stdin.nextLine())
                                      .map(Integer::new)
                                      .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        List<Integer> keys = d.keySet().stream().sorted().collect(Collectors.toList());
        int m = 0;
        for (int i = 0; i < keys.size(); i++) {
            Integer key = keys.get(i);
            m += d.get(key);
            if (n / 2 == m) {
                System.out.println(keys.get(i + 1) - (key + 1) + 1);
                System.exit(0);
            }
        }
        System.out.println(0);
    }

}

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.TreeMap;

public class Main {

    public static void main(String[] args) {
        @SuppressWarnings("resource")
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        String[] s = new String[n];
        for (int i = 0; i < n; i++) {
            s[i] = sc.next();
        }
        Map<String, Integer> map = new TreeMap<String, Integer>();
        for (int i = 0 ; i < n ; i++) {
            if (map.containsKey(s[i])) {
                map.put(s[i], map.get(s[i]) + 1);
            } else {
                map.put(s[i], 1);
            }
        }

        int min = 0;
        for (Entry<String, Integer> entry : map.entrySet() ) {
            min = Math.max(min, entry.getValue());
        }

        List<String> list = new ArrayList<String>();
        for (Entry<String, Integer> entry : map.entrySet() ) {
            if (entry.getValue() == min) {
                list.add(entry.getKey());
            }
        }

        Collections.sort(list);
        for (int i = 0 ; i < list.size() ; i++) {
            System.out.println(list.get(i));
        }


    }

}
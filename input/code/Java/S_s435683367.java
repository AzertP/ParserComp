import java.io.*;
import java.util.*;
public class Main {
    public static void main(String args[]) throws Exception {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int[][] intervals = new int[m][2];

        for(int i = 0; i < m; i++) {
          intervals[i][0] = sc.nextInt();
          intervals[i][1] = sc.nextInt();
        }
        Arrays.sort(intervals, (it1, it2) -> it1[1] - it2[1]);

        int i = 1;
        int cnt = 1;
        int endTime = intervals[0][1];
        while(i < m) {
          while(i < m && intervals[i][0] < endTime)
            i++;
          if(i < m) {
            cnt++;
            endTime = intervals[i][1];
          }
        }
        System.out.println(cnt);
    }
}

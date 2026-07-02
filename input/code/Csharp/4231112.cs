using System;
using System.Linq;

class Program {
    static void Main() {
        string[] stdin = Console.ReadLine().Split();
        int n = int.Parse(stdin[0]);
        int m = int.Parse(stdin[1]);
        int l = int.Parse(stdin[2]);
        long[][] a = new long[n][];
        for(int i = 0; i < n; i++) a[i] = Console.ReadLine().Split().Select(s => long.Parse(s)).ToArray();
        long[][] b = new long[m][];
        for(int i = 0; i < m; i++) b[i] = Console.ReadLine().Split().Select(s => long.Parse(s)).ToArray();
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < l; j++) {
                long ans = 0;
                for(int k = 0; k < m; k++) ans += a[i][k] * b[k][j];
                Console.Write("{0}{1}", ans, j == l - 1? '\n' : ' ');
            }
        }
    }
}

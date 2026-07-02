using System;
using System.Linq;

class Program {
    static void Main() {
        string[] stdin = Console.ReadLine().Split();
        int n = int.Parse(stdin[0]);
        int m = int.Parse(stdin[1]);
        int[][] a = new int[n][];
        for(int i = 0; i < n; i++) a[i] = Console.ReadLine().Split().Select(s => int.Parse(s)).ToArray();
        int[] b = new int[m];
        for(int i = 0; i < m; i++) b[i] = int.Parse(Console.ReadLine());
        for(int i = 0; i < n; i++) {
            int ans = 0;
            for(int j = 0; j < m; j++) ans += a[i][j] * b[j];
            Console.WriteLine(ans);
        }
    }
}

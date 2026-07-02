using System;
using System.Linq;
class Program {
    static int n, x;
    static int[] a;
    static bool[,] vis; // solve(depth, sum) ????¨??????????????????????????????°
    static bool[,] dp; // solve(depth, sum) ??????
    public static bool solve(int depth, int sum) {
        if (depth == n) return (sum == x);
        if (sum > x) return false; // sum ??? x ????¶???????????????§?????????
        if (vis[depth, sum]) return dp[depth, sum];
        bool ret = false;
        if (solve(depth + 1, sum + a[depth]) == true) ret = true; // a[depth] ????????¶??´???
        if (solve(depth + 1, sum) == true) ret = true; // a[depth] ????????°????????´???
        vis[depth, sum] = true;
        dp[depth, sum] = ret;
        return ret;
    }
    public static void Main() {
        n = int.Parse(Console.ReadLine());
        a = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int q = int.Parse(Console.ReadLine());
        int[] b = Console.ReadLine().Split().Select(int.Parse).ToArray();
        for (int i = 0; i < q; i++) {
            x = b[i];
            vis = new bool[n, x + 1];
            dp = new bool[n, x + 1];
            bool ret = solve(0, 0);
            Console.WriteLine(ret ? "yes" : "no");
        }
    }
}

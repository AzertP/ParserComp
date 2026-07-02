using System;

public class hello
{
    public static int n, m, L;
    public static int[,] a, b;
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        n = int.Parse(line[0]);
        m = int.Parse(line[1]);
        L = int.Parse(line[2]);
        a = getMatrix(n, m);
        b = getMatrix(m, L);
        getAns();
    }
    static void getAns()
    {
        for (int i = 0; i < n; i++)
        {
            var t = new long[L];
            for (int j = 0; j < L; j++)
                for (int k = 0; k < m; k++)
                    t[j] += (long)a[i, k] * b[k, j];
            Console.WriteLine(string.Join(" ", t));
        }
    }
    static int[,] getMatrix(int h, int w)
    {
        var ans = new int[h, w];
        for (int i = 0; i < h; i++)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            for (int j = 0; j < w; j++) ans[i, j] = int.Parse(line[j]);
        }
        return ans;
    }
}


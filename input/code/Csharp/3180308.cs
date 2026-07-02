using System;

public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var n = int.Parse(line[0]);
        var k = int.Parse(line[1]);
        var a = new int[n];
        for (int i = 0; i < n; i++) a[i] = int.Parse(Console.ReadLine().Trim());
        var ans = getAns(a, k);
        Console.WriteLine(ans);
    }
    static int getAns(int[] a , int k)
    {
        var ok = 1000000000;
        var ng = 0;
        while (ok - ng > 1)
        {
            var mid = ng + (ok - ng) / 2;
            if (check(a, k, mid)) ok = mid;
            else ng = mid;
        }
        return ok;
    }
    static bool check(int[] a,  int k, int b)
    {
        var aL = a.Length;
        var tcount = 0;
            var p = 0;
        for (int i = 0; i < aL; i++)
        {
            if (a[i] > b) return false;
            if (p + a[i] > b)
            {
                tcount++;
                if (tcount > k) return false;
                p = a[i];
            }
            else if (p + a[i] == b)
            {
                tcount++;
                if (tcount > k) return false;
                p = 0;
            }
            else p += a[i];
        }
        if (p > 0) tcount++;
        return tcount <= k;
    }
}



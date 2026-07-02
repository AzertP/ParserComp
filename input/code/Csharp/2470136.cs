using System;


public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var a = new int[n];

        string[] line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n; i++) a[i] = int.Parse(line[i]);
        var q = int.Parse(Console.ReadLine().Trim());
        string[] line2 = Console.ReadLine().Trim().Split(' ');
        
        for (int i = 0; i < q; i++)
        {
            var targ = int.Parse(line2[i]);
            if (Solve(a, 0, targ)) Console.WriteLine("yes");
            else Console.WriteLine("no");
        }

    }
public static bool Solve ( int[] a ,int i , int m  )
    {
        if (m == 0) return true;
        if (i >= a.Length) return false;
        var res = Solve(a,i + 1, m) | Solve(a,i + 1, m - a[i]);
        return res;
    }


}

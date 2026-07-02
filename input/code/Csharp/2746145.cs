using System;

public class Hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = Array.ConvertAll(line, int.Parse);
        var count = 0;
        for (int i = 0; i < n; i++)
        {
            var minj = i;
            for (int j = i; j < n; j++)
                if (a[j] < a[minj]) { minj = j; }
            if (i != minj) { count++; swap(a, i, minj); }
        }
        for (int i = 0; i < n; i++)
            if (i != n - 1) Console.Write("{0} ", a[i]);
            else Console.WriteLine(a[i]);
        Console.WriteLine(count);
    }
    public static void swap(int[] a, int i, int j)
    {
        var t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}


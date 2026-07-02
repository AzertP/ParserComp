using System;


public class hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        var ss = new int[n];
        string[] line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n; i++) ss[i] = int.Parse(line[i]);

        var n2 = int.Parse(Console.ReadLine().Trim());
        var tt = new int[n2];
        string[] line2 = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n2; i++) tt[i] = int.Parse(line2[i]);
        var count = 0;
        foreach (var x in tt)
            if (BS(ss,x)) count++;
        Console.WriteLine(count);

    }
    public static  bool  BS (int[] a  , int b)
    {
        var left = 0;
        var right = a.Length;
        while (left < right)
        {
            var mid = (left + right) / 2;
            if (a[mid] == b) return true;
            else if (b < a[mid]) right = mid;
            else left = mid + 1;
        }
        return false;
    }
}

using System.Linq;
using System.Collections.Generic;
using System;

public class hello
{
    public static void Main()
    {
        var s = new List<int>();
        var t = new List<int>();
        var n = int.Parse(Console.ReadLine().Trim());
        string[] line = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < n; i++)
            s.Add(int.Parse(line[i]));
        var q = int.Parse(Console.ReadLine().Trim());
        string[] line2 = Console.ReadLine().Trim().Split(' ');
        for (int i = 0; i < q; i++)
            t.Add(int.Parse(line2[i]));
        var result = 0;
        foreach (var x in t)
            if (s.Any(xx => xx == x)) result += 1;
        Console.WriteLine(result);
    }
}

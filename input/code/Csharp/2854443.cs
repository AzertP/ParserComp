using System;

public class Hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        string[] line = Console.ReadLine().Trim().Split(' ');
        string[] line2 = Console.ReadLine().Trim().Split(' ');
        var p1 = 0d; var p2 = 0d; var p3 = 0d;
        var wmax = 0d;
        for (int i = 0; i < n; i++)
        {
            var w = Math.Abs(double.Parse(line[i]) - double.Parse(line2[i]));
            wmax = Math.Max(wmax, w);
            p1 += w;
            var t = w * w;
            p2 += t;
            p3 += t * w;
        }
        Console.WriteLine(p1);
        Console.WriteLine(Math.Sqrt(p2));
        Console.WriteLine(Math.Pow(p3, 1 / 3d));
        Console.WriteLine(wmax);
    }
}


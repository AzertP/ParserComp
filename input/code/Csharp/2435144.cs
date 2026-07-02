using System;

public class hello
{

    public static void Main()
    {
        string[] s = Console.ReadLine().Trim().Split(' ');
        var x1 = double.Parse(s[0]);
        var y1 = double.Parse(s[1]);
        var x2 = double.Parse(s[2]);
        var y2 = double.Parse(s[3]);

        var buf = Math.Pow((y2 - y1), 2) + Math.Pow((x2 - x1), 2);
        var result = Math.Sqrt(buf);

        Console.WriteLine(result);
    }
}

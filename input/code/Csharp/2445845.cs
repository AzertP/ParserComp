using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = double.Parse(line[0]);
        var b = double.Parse(line[1]);
        var dosu = double.Parse(line[2]);
        var c = Math.PI * dosu / 180;
        var s = 0.5 * a * b * Math.Sin(c);
        Console.WriteLine(s);
        var buf = Math.Pow(a, 2d) + Math.Pow(b, 2d) - 2d * a * b * Math.Cos(c);
        Console.WriteLine(a + b + Math.Sqrt(buf));
        Console.WriteLine(b * Math.Sin(c));
    }
}

using System;


public class hello
{
    public static void Main()
    {
        var r = double.Parse(Console.ReadLine().Trim());
        var s = Math.Pow(r, 2) * Math.PI;
        var el = 2 * r * Math.PI;

        Console.WriteLine("{0} {1}", s, el);

    }
}

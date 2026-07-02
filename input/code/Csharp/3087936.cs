using System;

public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var a = int.Parse(line[0]);
        var b = int.Parse(line[1]);
        Console.WriteLine(gcd(a, b));
    }
    public static int gcd(int a, int b)
    {
        if (a < b) return gcd(b, a);
        while (b != 0)
        {
            var w = a % b;
            a = b;
            b = w;
        }
        return a;
    }
}

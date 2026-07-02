using System;

public class Program
{
    public static void Main()
    {
        double x = double.Parse(Console.ReadLine());

        double pii = 3.14159265359;
        double a = pii * (x * x);
        double c = 2 * pii * x;

        Console.WriteLine($"{a:F6} {c:F6}");
    }
}


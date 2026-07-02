using System;

public class Hello
{
    public static void Main()
    {
        var n = int.Parse(Console.ReadLine().Trim());
        for (int i = 3; i <= n; i++)
            if (i % 3 == 0 || i.ToString().Contains("3")) Console.Write(" " + i);
        Console.WriteLine();
    }
}


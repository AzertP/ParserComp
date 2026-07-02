using System;

public class hello
{
    public static void Main()
    {
        var count = 1;
        var n = 1;
        while (n != 0)
        {
            n = int.Parse(Console.ReadLine().Trim());
            if (n!=0)  Console.WriteLine("Case {0}: {1}", count, n);
            count += 1;
        }

    }
}

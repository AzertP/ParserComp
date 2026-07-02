using System;


public class hello
{
    public static void Main()
    {
        var buf   = Console.ReadLine().Trim();
        var  p = Console.ReadLine().Trim();
        var s = buf + buf;
        if (s.Contains(p)) Console.WriteLine("Yes");
        else Console.WriteLine("No");

    }
}

using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            var s = Console.ReadLine().Trim();
            if (s == "-") goto readend;
            var n = int.Parse(Console.ReadLine().Trim());
            for (int i = 0; i < n; i++)
            {
                var h = int.Parse(Console.ReadLine().Trim());
                s = s.Remove(0, h) + s.Substring(0, h);
            }
            Console.WriteLine(s);
        }
        readend:;
    }
}

using System;

public class Hello
{
    public static void Main()
    {
        while (true)
        {
            string[] line = Console.ReadLine().Trim().Split(' ');
            var n = int.Parse(line[0]);
            var x = int.Parse(line[1]);
            if (n == 0 && x == 0) break;
            var count = 0;
            for (int i = 1; i <= n; i++)
                for (int j = i + 1; j <= n; j++)
                {
                    var w = x - i - j;
                    if (w > j && w <= n) count++;
                }
            Console.WriteLine(count);
        }
    }
}


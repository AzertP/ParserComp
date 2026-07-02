using System;


public class hello
{
    public static void Main()
    {
        string[] line = Console.ReadLine().Trim().Split(' ');
        var n = int.Parse(line[0]);
        var m = int.Parse(line[1]);
        var a = new int[n, m];
        var b = new int[m];
        for (int i = 0; i < n; i++)
        {
            string[] line2 = Console.ReadLine().Trim().Split(' ');
            for (int j = 0; j < m; j++)
                a[i, j] = int.Parse(line2[j]);
        }
        for (int i = 0; i < m; i++)
            b[i] = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n; i++)
        {
            var total = 0;
            for (int j = 0; j < m; j++)
                total += a[i, j] * b[j];
            Console.WriteLine(total);
        }
    }
}

using System.Linq;
using System;

public class hello
{
    public static void Main()
    {
        while (true)
        {
            var n = int.Parse(Console.ReadLine().Trim());
            if (n == 0) goto readend;
            var ary = new double[n];
            string[] line = Console.ReadLine().Trim().Split(' ');
            for (int i = 0; i < n; i++)
                ary[i] = double.Parse(line[i]);
            var ave = ary.Average();
            for (int i = 0; i < n; i++)
                ary[i] = Math.Pow((ary[i] - ave), 2);
            var anser = Math.Sqrt(ary.Sum() / n);
            Console.WriteLine(anser);
        }
        readend:;
    }
}

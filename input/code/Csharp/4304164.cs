using System;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var tenant = new int[4, 3, 10];

            var n = int.Parse(Console.ReadLine());

            for (var i = 0; i < n; i++)
            {
                var line = Console.ReadLine().Split();
                var b = int.Parse(line[0]) - 1;
                var f = int.Parse(line[1]) - 1;
                var r = int.Parse(line[2]) - 1;
                var v = int.Parse(line[3]);

                tenant[b, f, r] += v;
            }

            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 3; j++)
                {
                    for (var k = 0; k < 10; k++)
                    {
                        Console.Write($" {tenant[i, j, k]}");
                    }
                    Console.WriteLine();
                }
                if (i != 3)
                {
                    Console.WriteLine("####################");
                }
            }
        }
    }
}


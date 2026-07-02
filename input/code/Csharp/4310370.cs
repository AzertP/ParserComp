using System;

namespace C_sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var line = Console.ReadLine();
            var rc = line.Split();
            var r = int.Parse(rc[0]);
            var c = int.Parse(rc[1]);

            var total = new int[c+1];
            for (var i = 0; i < r; i++)
            {
                line = Console.ReadLine();
                var row = line.Split(' ');

                var row_sum = 0;
                for (var j = 0; j < c; j++)
                {
                    var n = int.Parse(row[j]);
                    row_sum += n;
                    total[j] += n;
                }
                total[c] += row_sum;

                Console.Write(line);
                Console.WriteLine($" {row_sum}");
            }

            Console.WriteLine(string.Join(" ", total));
        }
    }
}

